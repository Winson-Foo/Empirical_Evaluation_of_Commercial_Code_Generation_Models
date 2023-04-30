from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
import os
from multiprocessing import Process, Queue, Event, Lock
from typing import Dict, Any, Optional, List

from .task_functions import create_process_task, create_process_task_gpu, task_executor_fn, task_executor_gpu_fn
from ..core.constants import BATCH, REALTIME, GPU, CPU, LOGGING_LEVEL, STOP_SIGNAL
from ..core.node import Node, ProducerNode, ConsumerNode, ProcessorNode
from ..core.task import Task, ProducerTask, ProcessorTask, ConsumerTask, MultiprocessingReceiveTask, MultiprocessingProcessorTask, MultiprocessingOutputTask
from ..core.engine import ExecutionEngine, Messenger
from ..utils.system import get_gpus_available_to_process

class BatchProcessingQueueMessenger(Messenger):
    '''
    BatchProcessingQueueMessenger is a messenger that communicates
    through queues of type ``multiprocessing.Queue``. It is not realtime,
    which means that if a queue is full when publishing a message to it,
    it will block until the queue can process it.
    '''

    def __init__(self, computation_node: Node, task_queue: Queue, parent_task_queue: Queue,
                 termination_event: Event):
        self.computation_node = computation_node
        self.parent_task_queue = parent_task_queue
        self.task_queue = task_queue
        self.parent_nodes_ids = []
        if self.computation_node.parents is not None:
            self.parent_nodes_ids = [parent.id for parent in self.computation_node.parents]
        self.termination_event = termination_event
        self.last_message_received = None
        self.logger = self._configure_logger()

    def _configure_logger(self):
        logger = logging.getLogger(f'{self.computation_node.id}')
        logger.setLevel(LOGGING_LEVEL)
        ch = logging.StreamHandler()
        ch.setLevel(LOGGING_LEVEL)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def publish_message(self, message: Any, metadata: Optional[Dict[str, Any]] = None):
        '''
        Publishes output message to a place where the child task will receive it.
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

    def publish_termination_message(self, message: Any, metadata: Optional[Dict[str, Any]] = None):
        '''
        This method is identical to publish_message.
        '''
        return self.publish_message(message, metadata)

    def passthrough_message(self):
        self.task_queue.put(self.last_message_received, block=True)

    def passthrough_termination_message(self):
        return self.passthrough_message()

    def receive_raw_message(self) -> Any:
        input_message_dict = self.parent_task_queue.get()
        self.last_message_received = input_message_dict
        self.logger.debug(f'Received message: {input_message_dict}')

        # 1. Check for STOP_SIGNAL before returning
        inputs = [input_message_dict[parent_node_id] for parent_node_id in self.parent_nodes_ids]
        messages = [input_message['message'] for input_message in inputs]
        stop_signal_received = any([isinstance(message, str) and message == STOP_SIGNAL for message in messages])

        # 2. Returns one or the other.
        if stop_signal_received:
            return STOP_SIGNAL
        else:
            return dict(input_message_dict)

    def receive_message(self) -> List[Any]:
        input_message_dict = self.parent_task_queue.get()
        self.last_message_received = input_message_dict
        self.logger.debug(f'Received message: {input_message_dict}')
        inputs = [input_message_dict[parent_node_id]['message'] for parent_node_id in self.parent_nodes_ids]
        return inputs

class BatchExecutionEngine(ExecutionEngine):

    def __init__(self):
        self.procs = []
        self.tasks = []
        self.task_output_queues: Dict[int, Queue] = {}
        self.task_termination_notification_queues: Dict[int, Queue] = {}
        self.termination_event = None
        self.gpu_ids = get_gpus_available_to_process()
        self.nb_available_gpus = len(self.gpu_ids)
        self.next_gpu_index = -1
        super(BatchExecutionEngine, self).__init__()

    def _create_output_queues(self, tasks_data: List[Tuple[Node, int, Optional[int], bool]]):
        for data in tasks_data:
            task_id = data[1]
            queue = Queue(1)
            self.task_output_queues[task_id] = queue

    def _initialize_tasks(self, tasks_data: List[Tuple[Node, int, Optional[int], bool]]):
        for (node, node_id, parent_node_id, is_last, *data_len) in tasks_data:
            is_multi_task_processor = isinstance(node, ProcessorNode) and node.nb_tasks > 1
            is_gpu_processor = isinstance(node, ProcessorNode) and node.device_type == GPU

            # Creating messenger for task
            task_queue = self.task_output_queues.get(node_id)
            if parent_node_id is not None:
                parent_task_queue = self.task_output_queues.get(parent_node_id)
            else:
                parent_task_queue = None

            messenger = BatchProcessingQueueMessenger(node, task_queue, parent_task_queue, self._termination_event)

            if isinstance(node, ProducerNode):
                task = ProducerTask(node, messenger, node_id, is_last)
                self.tasks.append(task)

            elif isinstance(node, ProcessorNode):
                if is_multi_task_processor:
                    receive_queue = Queue(1)
                    accounting_queue = Queue()
                    output_queues = [Queue() for _ in range(node.nb_tasks)]

                    # Create receive task
                    receive_task = MultiprocessingReceiveTask(
                        node,
                        parent_task_queue,
                        receive_queue,
                        BATCH
                    )
                    self.tasks.append(receive_task)

                    # Create processor tasks
                    mp_tasks_lock = Lock()
                    for idx in range(node.nb_tasks):
                        mp_task = MultiprocessingProcessorTask(
                            idx,
                            node,
                            mp_tasks_lock,
                            receive_queue,
                            accounting_queue,
                            output_queues[idx]
                        )
                        self.tasks.append(mp_task)

                    # Create output task
                    output_task = MultiprocessingOutputTask(
                        node,
                        task_queue,
                        accounting_queue,
                        output_queues,
                        BATCH,
                        is_last
                    )
                    self.tasks.append(output_task)
                elif is_gpu_processor:
                    if self.next_gpu_index < self.nb_available_gpus:
                        proc = create_process_task_gpu(node, messenger, node_id, is_last, self.gpu_ids[self.next_gpu_index])
                        self.procs.append(proc)
                        self.next_gpu_index += 1
                    else:
                        try:
                            node.change_device(CPU)
                            task = ProcessorTask(node, messenger, node_id, is_last, parent_node_id)
                            self.tasks.append(task)
                        except:
                            raise RuntimeError('No GPU available to allocate {}'.format(str(node._processor)))
                else:
                    task = ProcessorTask(node, messenger, node_id, is_last, parent_node_id)
                    self.tasks.append(task)

            elif isinstance(node, ConsumerNode):
                task = ConsumerTask(
                    node,
                    messenger,
                    node_id,
                    is_last,
                    parent_node_id
                )
                self.tasks.append(task)

    def _start_processes(self):
        for task in self.tasks:
            if isinstance(task, ProcessorTask) or isinstance(task, MultiprocessingProcessorTask):
                proc = create_process_task_gpu(task, self.gpu_ids[self.next_gpu_index])
                self.next_gpu_index += 1
            else:
                proc = create_process_task(task)
            self.procs.append(proc)

        for proc in self.procs:
            proc.start()

    def _al_create_and_start_processes(self, tasks_data: List[Tuple[Node, int, Optional[int], bool]]):
        self._create_output_queues(tasks_data)
        self._termination_event = Event()
        self._initialize_tasks(tasks_data)
        self._start_processes()

    def signal_flow_termination(self):
        self._termination_event.set()

    def join_task_processes(self):
        for proc in self.procs:
            try:
                proc.join()
            except KeyboardInterrupt:
                proc.join()
                continue