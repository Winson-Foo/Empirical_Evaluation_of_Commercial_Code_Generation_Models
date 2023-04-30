from multiprocessing import Process, Queue, Event, Lock
from typing import List, Dict, Any
import logging
import os

from ..core.constants import BATCH, REALTIME, GPU, CPU, LOGGING_LEVEL, STOP_SIGNAL
from ..core.engine import ExecutionEngine
from ..core.node import Node, ProducerNode, ConsumerNode, ProcessorNode
from ..core.task import Task, ProducerTask, ProcessorTask, ConsumerTask, MultiprocessingReceiveTask, MultiprocessingProcessorTask, MultiprocessingOutputTask
from ..utils.system import get_gpus_available_to_process

class BatchprocessingQueueMessenger:
    '''Batch processing messenger that communicates through multiprocessing queues.'''

    def __init__(self, computation_node: Node, task_queue: Queue, parent_task_queue: Queue,
                 termination_event: Event):
        self.computation_node = computation_node
        self.parent_task_queue = parent_task_queue
        self.task_queue = task_queue
        self.parent_nodes_ids = [parent.id for parent in self.computation_node.parents] if self.computation_node.parents else []
        self.termination_event = termination_event
        self.last_message_received = None
        self.logger = self._configure_logger()

    def _configure_logger(self):
        logger = logging.getLogger(self.computation_node.id)
        logger.setLevel(LOGGING_LEVEL)
        ch = logging.StreamHandler()
        ch.setLevel(LOGGING_LEVEL)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def publish_message(self, message: Any, metadata: Dict[str, Any] = None):
        '''
        Publishes output message to a place where the child task will receive it. \
        '''
        if self.last_message_received is None:
            msg = {
                self.computation_node.id: {
                    'message': message,
                    'metadata': metadata
                }
            }
            self.task_queue.put(msg, block=True)
            self.logger.debug(f'Published message {msg}')
        else:
            self.last_message_received[self.computation_node.id] = {
                'message': message,
                'metadata': metadata
            }
            self.task_queue.put(self.last_message_received, block=True)
            self.logger.debug(f'Published message {self.last_message_received}')

    def check_for_termination(self) -> bool:
        '''
        Checks if someone has set a termination event.
        '''
        return self.termination_event.is_set()

    def publish_termination_message(self, message: Any, metadata: Dict[str, Any] = None):
        '''
        This method is identical to publish message
        '''
        return self.publish_message(message, metadata)

    def passthrough_message(self):
        self.task_queue.put(self.last_message_received, block=True)

    def passthrough_termination_message(self):
        return self.passthrough_message()

    def receive_raw_message(self):
        input_message_dict = self.parent_task_queue.get()
        self.last_message_received = input_message_dict
        self.logger.debug(f'Received message: {input_message_dict}')

        # 1. Check for STOP_SIGNAL before returning
        inputs = [input_message_dict[a] for a in self.parent_nodes_ids]
        messages = [a['message'] for a in inputs]
        stop_signal_received = any([isinstance(a, str) and a == STOP_SIGNAL for a in messages])

        # 2. Returns one or the other.
        if stop_signal_received:
            return STOP_SIGNAL
        else:
            return dict(input_message_dict)

    def receive_message(self):
        input_message_dict = self.parent_task_queue.get()
        self.logger.debug(f'Received message: {input_message_dict}')
        self.last_message_received = input_message_dict
        inputs = [input_message_dict[a] for a in self.parent_nodes_ids]
        return inputs

class BatchExecutionEngine(ExecutionEngine):
    '''Batch execution engine for multiprocessing tasks.'''

    def __init__(self):
        self.procs = []
        self.tasks = []
        self.task_output_queues = {}
        self.task_termination_notification_queues = {}
        self.termination_event = None
        self.gpu_ids = get_gpus_available_to_process()
        self.nb_available_gpus = len(self.gpu_ids)
        self.next_gpu_index = -1
        super().__init__()

    def create_and_start_processes(self, tasks_data: List[Tuple[Node, str, str, bool, Any]]):
        '''Create and start multiprocessing processes for given tasks.'''
        #0. Create output queues
        for data in tasks_data:
            task_id = data[1]
            queue = Queue(1)
            self.task_output_queues[task_id] = queue

        self.termination_event = Event()

        #1. Initialize tasks
        tasks = []
        for (node, node_id, parent_node_id, is_last, *data_len) in tasks_data:
            # 1.1 Creating messenger for task
            task_queue = self.task_output_queues.get(node_id)
            parent_task_queue = self.task_output_queues.get(parent_node_id) if parent_node_id else None

            messenger = BatchprocessingQueueMessenger(node, task_queue, parent_task_queue, self.termination_event)

            if isinstance(node, ProducerNode):
                task = ProducerTask(node, messenger, node_id, is_last)
                tasks.append(task)

            elif isinstance(node, ProcessorNode):
                if node.nb_tasks > 1:
                    receiveQueue = Queue(1)
                    accountingQueue = Queue()
                    output_queues = [Queue() for _ in range(node.nb_tasks)]

                    # Create receive task
                    receive_task = MultiprocessingReceiveTask(
                        node,
                        parent_task_queue,
                        receiveQueue,
                        BATCH
                    )
                    tasks.append(receive_task)

                    # Create processor tasks
                    mp_tasks_lock = Lock()
                    for idx in range(node.nb_tasks):
                        mp_task = MultiprocessingProcessorTask(
                            idx,
                            node,
                            mp_tasks_lock,
                            receiveQueue,
                            accountingQueue,
                            output_queues[idx]
                        )
                        tasks.append(mp_task)

                    # Create output task
                    output_task = MultiprocessingOutputTask(
                        node,
                        task_queue,
                        accountingQueue,
                        output_queues,
                        BATCH,
                        is_last
                    )
                    tasks.append(output_task)
                else:
                    task = ProcessorTask(
                        node,
                        messenger,
                        node_id,
                        is_last,
                        parent_node_id
                    )
                    tasks.append(task)

            elif isinstance(node, ConsumerNode):
                task = ConsumerTask(
                    node,
                    messenger,
                    node_id,
                    is_last,
                    parent_node_id
                )
                tasks.append(task)

        #2. Create processes
        for task in tasks:
            if isinstance(task, ProcessorTask) or isinstance(task, MultiprocessingProcessorTask):
                if task.device_type == GPU:
                    self.next_gpu_index += 1
                    if self.next_gpu_index < self.nb_available_gpus:
                        proc = create_process_task_gpu(task, self.gpu_ids[self.next_gpu_index])
                    else:
                        try:
                            task.change_device(CPU)
                            proc = create_process_task(task)
                        except Exception as e:
                            raise RuntimeError(f'No GPU available to allocate {str(task._processor)}:\n {e}')
                else:
                    proc = create_process_task(task)
            else:
                proc = create_process_task(task)
            self.procs.append(proc)

        #3. Start processes.
        for proc in self.procs:
            proc.start()

    def signal_flow_termination(self):
        '''Set termination event to stop flow of tasks.'''
        self.termination_event.set()

    def join_task_processes(self):
        '''Join all processes associated with tasks.'''
        for proc in self.procs:
            try:
                proc.join()
            except KeyboardInterrupt:
                proc.join()
                continue

    def stop_task_processes(self):
        '''Terminate all processes associated with tasks.'''
        for proc in self.procs:
            proc.terminate()
            proc.join()

def create_process_task(task: Task) -> Process:
    '''Wrap task in a process.'''
    return Process(target=task_executor_fn, args=(task,))

def create_process_task_gpu(task: Task, gpu_id: int) -> Process:
    '''Wrap task in a process and assign it to a specific GPU.'''
    return Process(target=task_executor_gpu_fn, args=(task, gpu_id))