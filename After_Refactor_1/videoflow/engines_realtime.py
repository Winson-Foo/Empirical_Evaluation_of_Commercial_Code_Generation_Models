from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
import os
from multiprocessing import Process, Queue, Event, Lock

from .task_functions import create_process_task, create_process_task_gpu, task_executor_fn, task_executor_gpu_fn
from ..core.constants import BATCH, REALTIME, GPU, CPU, LOGGING_LEVEL, STOP_SIGNAL
from ..core.node import Node, ProducerNode, ConsumerNode, ProcessorNode
from ..core.task import Task, ProducerTask, ProcessorTask, ConsumerTask, MultiprocessingReceiveTask, MultiprocessingProcessorTask, MultiprocessingOutputTask
from ..core.engine import ExecutionEngine, Messenger
from ..utils.system import get_gpus_available_to_process

MAX_QUEUE_SIZE = 1

class RealtimeQueueMessenger(Messenger):
    '''
    RealtimeQueueMessenger is a messenger that communicates through
    queues of type ``multiprocessing.Queue``. It is real-time, which 
    means that if a queue is full when publishing a message to it,
    it will drop the message and not block. The methods that 
    publish and passthrough termination messages will block and not drop.
    '''
    def __init__(self, computation_node: Node, task_queue: Queue, parent_task_queue: Queue, termination_event: Event):
        self.node_id = computation_node.id
        self.computation_node = computation_node
        self.parent_nodes_ids = [parent.id for parent in computation_node.parents] if computation_node.parents else []
        self.task_queue = task_queue
        self.parent_task_queue = parent_task_queue
        self.termination_event = termination_event
        self.last_message_received = None
        self.logger = self._configure_logger()

    def _configure_logger(self):
        logger = logging.getLogger(f'{self.node_id}')
        logger.setLevel(LOGGING_LEVEL)
        ch = logging.StreamHandler()
        ch.setLevel(LOGGING_LEVEL)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def publish_message(self, message, metadata=None):
        '''
        Publishes output message to a place where the child task will receive it. \
        Will drop the message if the receiving queue is full.
        '''
        if self.last_message_received is None:
            try:
                msg = {
                    self.node_id: {
                        'message': message,
                        'metadata': metadata
                    }
                }
                self.task_queue.put(msg, block=False)
                self.logger.debug(f'Published message {msg}')
            except:
                self.logger.debug(f'Queue is full.')
        else:
            self.last_message_received[self.node_id] = {
                'message': message,
                'metadata': metadata
            }
            try:
                self.task_queue.put(self.last_message_received, block=False)
                self.logger.debug(f'Published message {self.last_message_received}')
            except:
                self.logger.debug(f'Queue is full.')

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
                    self.node_id: {
                        'message': message,
                        'metadata': metadata
                    }
                }
                self.task_queue.put(msg, block=True)
            except:
                pass
        else:
            self.last_message_received[self.node_id] = {
                'message': message,
                'metadata': metadata
            }
            try:
                self.task_queue.put(self.last_message_received, block=True)
            except:
                pass

    def passthrough_message(self):
        try:
            self.task_queue.put(self.last_message_received, block=False)
        except:
            pass
    
    def passthrough_termination_message(self):
        try:
            self.task_queue.put(self.last_message_received, block=True)
        except:
            pass

    def receive_message(self):
        input_message_dict = self.parent_task_queue.get()
        self.logger.debug(f'Received message: {input_message_dict}')
        self.last_message_received = input_message_dict
        inputs = [input_message_dict.get(parent_id) for parent_id in self.parent_nodes_ids]
        return inputs


class RealtimeExecutionEngine(ExecutionEngine):
    def __init__(self):
        self.processes = []
        self.tasks = []
        self.task_output_queues = {}
        self.task_termination_notification_queues = {}
        self.termination_event = None
        self.gpu_ids = get_gpus_available_to_process()
        self.nb_available_gpus = len(self.gpu_ids)
        self.next_gpu_index = -1
        super(RealtimeExecutionEngine, self).__init__()

    def _al_create_processes(self, tasks_data):
        #0. Create output queues
        for computation_node, node_id, parent_node_id, is_last in tasks_data:
            self.task_output_queues[node_id] = Queue(MAX_QUEUE_SIZE)

        self.termination_event = Event()

        #1. Initialize tasks
        for computation_node, node_id, parent_node_id, is_last in tasks_data:
            task_queue = self.task_output_queues.get(node_id)
            parent_task_queue = self.task_output_queues.get(parent_node_id) if parent_node_id is not None else None
            messenger = RealtimeQueueMessenger(computation_node, task_queue, parent_task_queue, self.termination_event)

            if isinstance(computation_node, ProducerNode):
                task = ProducerTask(computation_node, messenger, node_id, is_last)
                self.tasks.append(task)

            elif isinstance(computation_node, ProcessorNode):
                if computation_node.nb_tasks > 1:
                    receive_queue = Queue(MAX_QUEUE_SIZE)
                    accounting_queue = Queue()
                    output_queues = [Queue(MAX_QUEUE_SIZE) for _ in range(computation_node.nb_tasks)]
                    receive_task = MultiprocessingReceiveTask(computation_node, parent_task_queue, receive_queue, REALTIME)
                    self.tasks.append(receive_task)
                    mp_tasks_lock = Lock()
                    for idx in range(computation_node.nb_tasks):
                        mp_task = MultiprocessingProcessorTask(idx, computation_node, mp_tasks_lock, receive_queue, accounting_queue,output_queues[idx])
                        self.tasks.append(mp_task)
                    output_task = MultiprocessingOutputTask(computation_node, task_queue, accounting_queue, output_queues, REALTIME, is_last)
                    self.tasks.append(output_task)
                else:
                    task = ProcessorTask(computation_node, messenger, node_id, is_last, parent_node_id)
                    self.tasks.append(task)

            elif isinstance(computation_node, ConsumerNode):
                task = ConsumerTask(computation_node, messenger, node_id, is_last, parent_node_id)
                self.tasks.append(task)

        #2. Create processes
        for task in self.tasks:
            if isinstance(task, ProcessorTask) or isinstance(task, MultiprocessingProcessorTask):
                if task.device_type == GPU:
                    self.next_gpu_index += 1

                    try:
                        gpu_id = self.gpu_ids[self.next_gpu_index]
                    except:
                        task.change_device(CPU)
                        gpu_id = None

                    proc = create_process_task_gpu(task, gpu_id)
                else:
                    proc = create_process_task(task)
            else:
                proc = create_process_task(task)
            
            self.processes.append(proc)

    def _al_start_processes(self):
        #3. Start processes.
        for proc in self.processes:
            proc.start()

    def _al_create_and_start_processes(self, tasks_data):
        self._al_create_processes(tasks_data)
        self._al_start_processes()

    def signal_flow_termination(self):
        self.termination_event.set()

    def join_task_processes(self):
        for proc in self.processes:
            try:
                proc.join()
            except KeyboardInterrupt:
                proc.join()
                continue
