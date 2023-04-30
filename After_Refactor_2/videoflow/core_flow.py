from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging

from .graph import GraphEngine
from .constants import BATCH, REALTIME, FLOW_TYPES, STOP_SIGNAL
from .node import Node, ProducerNode, ConsumerNode, ProcessorNode
from .task import Task, ProducerTask, ProcessorTask, ConsumerTask
from .bottlenecks import MetadataConsumer
from ..engines.realtime import RealtimeExecutionEngine
from ..engines.batch import BatchExecutionEngine

logger = logging.getLogger(__package__)

def get_task_data_from_topological_sort(tsort_l):
    '''
    Returns a list of task data generated from the given topological sort
    of nodes in the computation graph.
    '''
    tasks_data = []

    for i, node in enumerate(tsort_l):
        if isinstance(node, ProducerNode):
            task_data = (node, i, None, i >= (len(tsort_l) - 1))
        elif isinstance(node, ProcessorNode) or isinstance(node, ConsumerNode):
            task_data = (node, i, i - 1, i >= (len(tsort_l) - 1))
        else:
            raise ValueError('Node is not of one of the valid types')
        tasks_data.append(task_data)
        
    return tasks_data

class Flow:
    '''
    Represents a linear flow of data from one task to another.\
    Note that a flow is created from a **directed acyclic graph** of producer, processor \
    and consumer nodes, but the flow itself is **linear**, because it is an optimized \
    `topological sort` of the directed acyclic graph.

    - Arguments:
        - producers: a list of producer nodes of type ``videoflow.core.node.ProducerNode``.
        - consumers: a list of consumer nodes of type ``videoflow.core.node.ConsumerNode``.
        - flow_type: one of 'realtime' or 'batch'
    '''
    def __init__(self, producers, consumers, flow_type=REALTIME):
        self.graph_engine = GraphEngine(producers, consumers)
        
        if flow_type not in FLOW_TYPES:
            raise ValueError('flow_type must be one of {}'.format(','.join(FLOW_TYPES)))
        
        if flow_type == BATCH:
            self.execution_engine = BatchExecutionEngine()
        elif flow_type == REALTIME:
            self.execution_engine = RealtimeExecutionEngine()

    def run(self):
        '''
        Starts the flow by creating a topological sort of the nodes in the computation graph, 
        wrapping each node around a ``videoflow.core.task.Task`` and passing the tasks to 
        the environment which will allocate them and create the channels for communication 
        between tasks.
        '''
        # Build a topological sort of the graph including a metadata consumer at the end.
        tsort = self.graph_engine.topological_sort() + [MetadataConsumer()]
        
        # TODO: Optimize the graph
        
        # Create the tasks and their input/outputs.
        tasks_data = get_task_data_from_topological_sort(tsort)
        
        # Start running the flow.
        self.execution_engine.allocate_and_run_tasks(tasks_data)
        logger.info('Allocated processes for {} tasks'.format(len(tasks_data)))
        logger.info('Started running flow.')
    
    def join(self):
        '''
        Blocks the process that calls this method until the flow finishes running naturally.
        '''
        self.execution_engine.join_task_processes()
        logger.info('Flow has stopped.')

    def stop(self):
        '''
        Causes the execution environment to send a flow termination signal and blocks 
        until the flow stops running.
        '''
        logger.info('Stop termination signal placed on flow.')
        self.execution_engine.signal_flow_termination()
        self.join()