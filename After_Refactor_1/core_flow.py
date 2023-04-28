from __future__ import absolute_import, division, print_function

from typing import List

from videoflow.core.bottlenecks import MetadataConsumer
from videoflow.core.constants import BATCH, FLOW_TYPES, REALTIME, STOP_SIGNAL
from videoflow.core.engines.batch import BatchExecutionEngine
from videoflow.core.engines.realtime import RealtimeExecutionEngine
from videoflow.core.graph import GraphEngine
from videoflow.core.node import (
    ConsumerNode,
    Node,
    ProcessorNode,
    ProducerNode,
)
from videoflow.core.task import (
    ConsumerTask,
    ProcessorTask,
    ProducerTask,
    Task,
)

import logging

logger = logging.getLogger(__package__)


class Flow:
    """
    Represents a linear flow of data from one task to another.
    Note that a flow is created from a **directed acyclic graph** of producer, processor
    and consumer nodes, but the flow itself is **linear**, because it is an optimized
    `topological sort` of the directed acyclic graph.

    :param producers: A list of producer nodes.
    :param consumers: A list of consumer nodes.
    :param flow_type: One of 'realtime' or 'batch'.
    """

    def __init__(
        self,
        producers: List[ProducerNode],
        consumers: List[ConsumerNode],
        flow_type: str = REALTIME,
    ):
        if flow_type not in FLOW_TYPES:
            raise ValueError(f"flow_type must be one of {FLOW_TYPES}")
        self._graph_engine = GraphEngine(producers, consumers)
        self._tasks_data = None
        if flow_type == BATCH:
            self._execution_engine = BatchExecutionEngine()
        elif flow_type == REALTIME:
            self._execution_engine = RealtimeExecutionEngine()

    def run(self):
        """
        Starts the flow.
        """
        self._build_tasks_data()
        self._execution_engine.allocate_and_run_tasks(self._tasks_data)
        logger.info(f"Allocated processes for {len(self._tasks_data)} tasks")
        logger.info("Started running flow")

    def join(self):
        """
        Makes the process that calls this method block until the flow finishes running naturally.
        """
        self._execution_engine.join_task_processes()
        logger.info("Flow has stopped")

    def stop(self):
        """
        Stops the flow. Makes the execution environment send a flow termination signal.
        """
        logger.info("Stop termination signal placed on flow")
        self._execution_engine.signal_flow_termination()
        self.join()

    def _task_data_from_node_tsort(self, tsort_l):
        """
        Converts a topological sort of the graph into a list of TaskData
        """
        tasks_data = []

        for i in range(len(tsort_l)):
            node = tsort_l[i]
            if isinstance(node, ProducerNode):
                task_data = (node, i, None, i >= (len(tsort_l) - 1))
            elif isinstance(node, ProcessorNode):
                task_data = (node, i, i - 1, i >= (len(tsort_l) - 1))
            elif isinstance(node, ConsumerNode):
                task_data = (node, i, i - 1, i >= (len(tsort_l) - 1))
            else:
                raise ValueError("node is not of one of the valid types")
            tasks_data.append(task_data)

        return tasks_data

    def _build_tasks_data(self):
        """
        Builds a topological sort of the graph and converts it into a list of TaskData.
        """
        tsort = self._graph_engine.topological_sort()
        metadata_consumer = MetadataConsumer()(*tsort)
        tsort.append(metadata_consumer)
        self._tasks_data = self._task_data_from_node_tsort(tsort)
