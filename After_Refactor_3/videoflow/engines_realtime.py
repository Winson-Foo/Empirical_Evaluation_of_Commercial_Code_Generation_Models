from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging

import os
from multiprocessing import Process, Queue, Event, Lock
from typing import Tuple, List

from .task_functions import create_process_task, create_process_task_gpu, task_executor_fn, task_executor_gpu_fn
from ..core.constants import BATCH, REALTIME, GPU, CPU, LOGGING_LEVEL, STOP_SIGNAL
from ..core.node import Node, ProducerNode, ConsumerNode, ProcessorNode
from ..core.task import Task, ProducerTask, ProcessorTask, ConsumerTask, MultiprocessingReceiveTask, MultiprocessingProcessorTask, MultiprocessingOutputTask
from ..core.engine import ExecutionEngine, Messenger
from ..utils.system import get_gpus_available_to_process

class RealtimeQueueMessenger(Messenger):
    '''
    RealtimeQueueMessenger is a messenger that communicates through
    queues of type ``multiprocessing.Queue``.  It is a realtime messenger, 
    which means that if a queue is full when publishing a message to it,
    it will drop the message and not block.  The methods that 
    publish and passthrough termination messages will block and not drop.
    '''
    def __init__(self, computation_node: Node, task_queue: Queue, parent_task_queue: Queue,
                 termination_event: Event):
        self._computation_node = computation_node
        self._parent_task_queue = parent_task_queue
        self._task_queue = task_queue
        self._parent_nodes_ids = [p.id for p in (self._computation_node.parents or [])]
        self._termination_event = termination_event
        self._last_message_received = None
        self._logger = self._configure_logger()

    def _configure_logger(self):
        logger = logging.getLogger(f'{self._computation_node.id}')
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
        try:
            msg = {self._computation_node.id: {'message': message, 'metadata': metadata}}
            self._task_queue.put(msg, block=False)
            self._logger.debug(f'Published message {msg}')
        except Queue.Full:
            self._logger.debug(f'Queue is full.')
            pass

    def check_for_termination(self) -> bool:
        '''
        Checks if someone has set a termination event.
        '''
        return self._termination_event.is_set()

    def publish_termination_message(self, message, metadata=None):
        '''
        This method is identical to publish message, but is blocking
        Because, the termination message cannot be dropped.
        '''
        try:
            msg = {self._computation_node.id: {'message': message, 'metadata': metadata}}
            self._task_queue.put(msg, block=True)
        except Queue.Full:
            pass

    def passthrough_message(self):
        try:
            self._task_queue.put(self._last_message_received, block=False)
        except Queue.Full:
            pass
    
    def passthrough_termination_message(self):
        try:
            self._task_queue.put(self._last_message_received, block=True)
        except Queue.Full:
            pass

    def receive_message(self) -> List:
        input_message_dict = self._parent_task_queue.get()
        self._logger.debug(f'Received message: {input_message_dict}')
        self._last_message_received = input_message_dict
        inputs = [input_message_dict[a] for a in self._parent_nodes_ids]
        return inputs

class RealtimeExecutionEngine(ExecutionEngine):
    def __init__(self):
        self._procs = []
        self._tasks = []
        self._task_output_queues = {}
        self._task_termination_notification_queues = {}
        self._termination_event = None
        self._gpu_ids = get_gpus_available_to_process()
        self._nb_available_gpus = len(self._gpu_ids)
        self._next_gpu_index = -1
        super(RealtimeExecutionEngine, self).__init__()
        
    def _al_create_processes(self, tasks_data: List[Tuple[Node, int, int, bool]]):
        '''
        Create output queues
        '''
        for data in tasks_data:
            task_id = data[1]
            queue = Queue(1)
            self._task_output_queues[task_id] = queue
        
        self._termination_event = Event()

        # Initialize tasks
        tasks = []
        for data in tasks_data:
            node = data[0]
            node_id = data[1]
            parent_node_id = data[2]
            is_last = data[3]

            # Create messenger for task
            task_queue = self._task_output_queues.get(node_id)
            parent_task_queue = self._task_output_queues.get(parent_node_id) if parent_node_id is not None else None
        
            messenger = RealtimeQueueMessenger(node, task_queue, parent_task_queue, self._termination_event)

            if isinstance(node, ProducerNode):
                task = ProducerTask(node, messenger, node_id, is_last)
                tasks.append(task)

            elif isinstance(node, ProcessorNode):
                if node.nb_tasks > 1:
                    receiveQueue = Queue(1)
                    accountingQueue = Queue()
                    output_queues = [Queue() for _ in range(node.nb_tasks)]

                    # Create receive task
                    receive_task = MultiprocessingReceiveTask(node, parent_task_queue, receiveQueue, REALTIME)
                    tasks.append(receive_task)

                    # Create processor tasks
                    mp_tasks_lock = Lock()
                    for idx in range(node.nb_tasks):
                        mp_task = MultiprocessingProcessorTask(idx, node, mp_tasks_lock, receiveQueue,
                                                               accountingQueue, output_queues[idx])
                        tasks.append(mp_task)
                    
                    # Create output task
                    output_task = MultiprocessingOutputTask(node, task_queue, accountingQueue, output_queues, REALTIME,
                                                             is_last)
                    tasks.append(output_task)
                else:
                    task = ProcessorTask(node, messenger, node_id, is_last, parent_node_id)
                    tasks.append(task)

            elif isinstance(node, ConsumerNode):
                task = ConsumerTask(node, messenger, node_id, is_last, parent_node_id)
                tasks.append(task)
        
        # Create processes
        for task in tasks:
            if isinstance(task, ProcessorTask) or isinstance(task, MultiprocessingProcessorTask):
                if task.device_type == GPU:
                    self._next_gpu_index += 1
                    if self._next_gpu_index < self._nb_available_gpus:
                        proc = create_process_task_gpu(task, self._gpu_ids[self._next_gpu_index])
                    else:
                        try:
                            task.change_device(CPU)
                            proc = create_process_task(task)
                        except:
                            raise RuntimeError(f'No GPU available to allocate {task._processor}')
                else:
                    proc = create_process_task(task)
            else:
                proc = create_process_task(task)
            self._procs.append(proc)

    def _al_start_processes(self):
        # Start processes.
        for proc in self._procs:
            proc.start()

    def _al_create_and_start_processes(self, tasks_data: List[Tuple[Node, int, int, bool]]):
        self._al_create_processes(tasks_data)
        self._al_start_processes()
        
    def signal_flow_termination(self):
        self._termination_event.set()
    
    def join_task_processes(self):
        for proc in self._procs:
            try:
                proc.join()
            except KeyboardInterrupt:
                proc.join()
                continue


# Sample usage:
tasks_data = []  # List of tuples containing (Node, task_id, parent_node_id, is_last) information
engine = RealtimeExecutionEngine()
engine._al_create_and_start_processes(tasks_data)
engine.join_task_processes()