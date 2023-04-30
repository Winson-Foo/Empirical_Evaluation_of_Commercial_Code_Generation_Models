from ..utils.graph import has_cycle, topological_sort
from .node import ProducerNode

import logging

LOGGER = logging.getLogger(__package__)
ONLY_ONE_PRODUCER_SUPPORTED_ERROR = 'Only support flows with 1 producer for now.'
INVALID_PRODUCER_NODE_ERROR = '{} is not instance of ProducerNode'
CYCLE_DETECTED_ERROR = 'Cycle detected in computation graph. Exiting now...'
CYCLE_FOUND_ERROR = 'Cycle found in graph'
CONSUMER_NOT_DESCENDANT_ERROR = (
    'Consumer {consumer} is not descendant of any producer. Exiting now...'
)

class GraphEngine:
    """
    Engine for executing a computation graph
    """

    def __init__(self, producers, consumers):
        """
        Initializes a new GraphEngine instance with the specified producers and consumers.

        :param producers: A list of ProducerNode instances, with only one producer supported at the moment.
        :param consumers: A list of Node instances that consume the output of the producers.
        """
        if len(producers) != 1:
            raise AttributeError(ONLY_ONE_PRODUCER_SUPPORTED_ERROR)

        for producer in producers:
            if not isinstance(producer, ProducerNode):
                raise AttributeError(INVALID_PRODUCER_NODE_ERROR.format(producer))

        self.producers = producers
        self.consumers = consumers

        if has_cycle(self.producers):
            LOGGER.error(CYCLE_DETECTED_ERROR)
            raise ValueError(CYCLE_FOUND_ERROR)

        self.sorted_producers = topological_sort(self.producers)
        LOGGER.debug(f"Topological sort: {self.sorted_producers}")

        for consumer in consumers:
            if consumer not in self.sorted_producers:
                LOGGER.error(CONSUMER_NOT_DESCENDANT_ERROR.format(consumer=consumer))
                raise ValueError(f"{consumer} is not descendant of any producer")

        # TODO: Check that all producers' results are being read by a consumer.

    def topological_sort(self):
        """
        Returns a list containing a topologically sorted order of the producers in the computation graph.

        :return: A list containing ProducerNode instances in a topologically sorted order.
        """
        return list(self.sorted_producers)