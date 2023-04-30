from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
from multiprocessing import Event, Lock, Process, Queue

from ..core.constants import BATCH, CPU, GPU, LOGGING_LEVEL, REALTIME, STOP_SIGNAL
from ..core.engine import ExecutionEngine, Messenger
from ..core.node import ConsumerNode, Node, ProcessorNode, ProducerNode
from ..core.task import ConsumerTask, MultiprocessingOutputTask, MultiprocessingProcessorTask, MultiprocessingReceiveTask, ProcessorTask, ProducerTask, Task
from ..utils.system import get_gpus_available_to_process


class RealtimeQueueMessenger(Messenger):
    '''
    RealtimeQueueMessenger is a messenger that communicates through
    queues of type ``multiprocessing.Queue``. It is a real time, which
    means that if a queue is full when publishing a message to it,
    it will drop the message and not block. The methods that
    publish and pass through termination messages will block and not drop.
    '''

    def __init__(self, computation_node: Node, task_queue: Queue, parent_task_queue: Queue,
                 termination_event: Event):
        self.computation_node = computation_node
        self.parent_task_queue = parent_task_queue
        self.task_queue = task_queue
        self.parent_nodes_ids = []

        if self.computation_node.parents is not None:
            self.parent_nodes_ids = [parent_node.id for parent_node in self.computation_node.parents]

        self.termination_event = termination_event
        self.last_message_received = None
        self.logger = self.configure_logger()

    def configure_logger(self):
        logger = logging.getLogger(f'{self.computation_node.id}')
        logger.setLevel(LOGGING_LEVEL)
        ch = logging.StreamHandler()
        ch.setLevel(LOGGING_LEVEL)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        return logger

    def publish_message(self, message, metadata=None):
        '''
        Publishes output message to a place where the child task will receive it.
        Will drop the message if the receiving queue is full.
        '''
        if self.last_message_received is None:
            try:
                msg = {
                    self.computation_node.id: {
                        'message': message,
                        'metadata': metadata
                    }
                }

                self.task_queue.put(msg, block=False)
                self.logger.debug(f'Published message {msg}')
            except:
                self.logger.debug(f'Queue is full.')
                pass
        else:
            self.last_message_received[self.computation_node.id] = {
                'message': message,
                'metadata': metadata
            }
            try:
                self.task_queue.put(self.last_message_received, block=False)
                self.logger.debug(f'Published message {self.last_message_received}')
            except:
                self.logger.debug(f'Queue is full.')
                pass

    def check_for_termination(self) -> bool:
        '''
        Checks if someone has set a termination event.
        '''
        return self.termination_event.is_set()

    def publish_termination_message(self, message, metadata=None):
        '''
        This method is identical to publish message, but is blocking
        Because, the termination message cannot be dropped.
        '''
        if self.last_message_received is None:
            try:
                msg = {
                    self.computation_node.id: {
                        'message': message,
                        'metadata': metadata
                    }
                }

                self.task_queue.put(msg, block=True)
            except:
                pass
        else:
            self.last_message_received[self.computation_node.id] = {
                'message': message,
                'metadata': metadata
            }

            try:
                self.task_queue.put(self.last_message_received, block=True)
            except:
                pass

    def pass_through_message(self):
        try:
            self.task_queue.put(self.last_message_received, block=False)
        except:
            pass

    def pass_through_termination_message(self):
        try:
            self.task_queue.put(self.last_message_received, block=True)
        except:
            pass

    def receive_message(self):
        input_message_dict = self.parent_task_queue.get()
        self.logger.debug(f'Received message: {input_message_dict}')
        self.last_message_received = input_message_dict
        inputs = [input_message_dict[parent_node_id] for parent_node_id in self.parent_nodes_ids]

        return inputs


class RealtimeExecutionEngine(ExecutionEngine):
    def __init__(self):
        self.procs = []
        self.tasks = []
        self.task_output_queues = {}
        self.task_termination_notification_queues = {}
        self.termination_event = None
        self.gpu_ids = get_gpus_available_to_process()
        self.nb_available_gpus = len(self.gpu_ids)
        self.next_gpu_index = -1
        super(RealtimeExecutionEngine, self).__init__()

    def prepare_processes(self, tasks_data):
        # 0. Create output queues
        for data in tasks_data:
            task_id = data[1]
            queue = Queue(1)
            self.task_output_queues[task_id] = queue

        self.termination_event = Event()

        # 1. Initialize tasks
        tasks = []
        for data in tasks_data:
            node = data[0]
            node_id = data[1]
            parent_node_id = data[2]
            is_last = data[3]

            # 1.1 Creating messenger for task
            task_queue = self.task_output_queues.get(node_id)
            if parent_node_id is not None:
                parent_task_queue = self.task_output_queues.get(parent_node_id)
            else:
                parent_task_queue = None

            messenger = RealtimeQueueMessenger(node, task_queue, parent_task_queue, self.termination_event)

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
                        REALTIME
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
                        REALTIME,
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

        # 2. Create processes
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
                        except:
                            raise RuntimeError('No GPU available to allocate {}'.format(str(task._processor)))
                else:
                    proc = create_process_task(task)
            else:
                proc = create_process_task(task)

            self.procs.append(proc)

    def start_processes(self):
        for proc in self.procs:
            proc.start()

    def create_and_start_processes(self, tasks_data):
        self.prepare_processes(tasks_data)
        self.start_processes()

    def signal_flow_termination(self):
        self.termination_event.set()

    def join_task_processes(self):
        for proc in self.procs:
            try:
                proc.join()
            except KeyboardInterrupt:
                proc.join()
                continue