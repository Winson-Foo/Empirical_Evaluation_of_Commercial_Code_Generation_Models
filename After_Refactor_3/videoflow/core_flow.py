from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging

from videoflow.core.graph import GraphEngine
from videoflow.core.constants import BATCH, REALTIME, FLOW_TYPES
from videoflow.core.node import Node, ProducerNode, ConsumerNode, ProcessorNode
from videoflow.core.task import Task, ProducerTask, ProcessorTask, ConsumerTask
from videoflow.core.bottlenecks import MetadataConsumer
from videoflow.engines.realtime import RealtimeExecutionEngine
from videoflow.engines.batch import BatchExecutionEngine

logger = logging.getLogger(__package__)


class Flow:
    """
    Represents a linear flow of data from one task to another.\
    Note that a flow is created from a **directed acyclic graph** of producer, \
    processor and consumer nodes, but the flow itself is **linear**, because \
    it is an optimized `topological sort` of the directed acyclic graph.

    :param producers: a list of producer nodes of type videoflow.core.node.ProducerNode.
    :param consumers: a list of consumer nodes of type videoflow.core.node.ConsumerNode.
    :param flow_type: one of 'realtime' or 'batch'
    """

    def __init__(self, producers, consumers, flow_type=REALTIME):
        """
        Initialize the Flow object.

        :param producers: a list of producer nodes.
        :param consumers: a list of consumer nodes.
        :param flow_type: a string that indicates the flow type (batch or realtime).
        """
        self.graph_engine = GraphEngine(producers, consumers)
        if flow_type not in FLOW_TYPES:
            raise ValueError(f"flow_type must be one of {','.join(FLOW_TYPES)}")
        if flow_type == BATCH:
            self.execution_engine = BatchExecutionEngine()
        elif flow_type == REALTIME:
            self.execution_engine = RealtimeExecutionEngine()

    def run(self):
        """
        Start the flow, creating tasks from the nodes in the computation graph, \
        and passing the tasks to the environment, which allocates them and creates \
        the channels that will be used for communication between tasks.
        """
        # Build a topological sort of the graph
        tsort = self.graph_engine.topological_sort()
        metadata_consumer = MetadataConsumer()(*tsort)
        tsort.append(metadata_consumer)

        # Create the tasks and the input/outputs for them
        tasks_data = []

        task_data_map = {
            ProducerNode: lambda node, i: (node, i, None, i >= (len(tsort) - 1)),
            ProcessorNode: lambda node, i: (node, i, i - 1, i >= (len(tsort) - 1)),
            ConsumerNode: lambda node, i: (node, i, i - 1, i >= (len(tsort) - 1)),
        }

        for i, node in enumerate(tsort):
            if type(node) not in task_data_map:
                raise ValueError("node is not of one of the valid types")
            task_data = task_data_map[type(node)](node, i)
            tasks_data.append(task_data)

        # Put each task to run in the place where the processor it contains inside runs
        self.execution_engine.allocate_and_run_tasks(tasks_data)
        logger.info(f"Allocated processes for {len(tasks_data)} tasks")
        logger.info("Started running flow.")

    def join(self):
        """
        Block until the flow finishes running naturally.
        """
        self.execution_engine.join_task_processes()
        logger.info("Flow has stopped.")

    def stop(self):
        """
        Stop the flow.
        """
        logger.info("Stop termination signal placed on flow.")
        self.execution_engine.signal_flow_termination()
        self.join()