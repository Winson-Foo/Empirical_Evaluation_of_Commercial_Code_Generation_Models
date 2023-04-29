from typing import List
from ..utils.graph import has_cycle, topological_sort
from .node import ProducerNode

import logging
logger = logging.getLogger(__package__)

class GraphEngine:
    """
    Class representing a graph engine used to process computation graphs.

    Attributes:
        producers (List[ProducerNode]): List of ProducerNode objects representing the graph's producers.
        consumers (List[Node]): List of Node objects representing the graph's consumers.
        tsort (List[Node]): List of Node objects representing the graph's topological sort.
    """

    def __init__(self, producers: List[ProducerNode], consumers: List[Node]) -> None:
        """
        Initializes a new GraphEngine object.

        Args:
            producers (List[ProducerNode]): List of ProducerNode objects representing the graph's producers.
            consumers (List[Node]): List of Node objects representing the graph's consumers.

        Raises:
            AttributeError: Raised if the number of producers is not 1 or if any producer in the list is not an instance of ProducerNode.
            ValueError: Raised if a cycle is detected in the computation graph or if any consumer is not a descendant of any of the producers.
        """
        if len(producers) != 1:
            raise AttributeError('Only support flows with 1 producer for now.')

        for producer in producers:
            if not isinstance(producer, ProducerNode):
                raise AttributeError('{} is not instance of ProducerNode'.format(producer))

        self.producers = producers
        self.consumers = consumers

        if has_cycle(self.producers):
            logger.error('Cycle detected in computation graph. Exiting now...')
            raise ValueError('Cycle found in graph')

        self.tsort = topological_sort(self.producers)
        logger.debug("Topological sort: {}".format(self.tsort))

        for consumer in consumers:
            if consumer not in self.tsort:
                logger.error(f'Consumer {consumer} is not descendant of any producer. Exiting now...')
                raise ValueError(f'{consumer} is not descendant of any producer')

    def topological_sort(self) -> List[Node]:
        """
        Returns the graph's topological sort.

        Returns:
            List[Node]: List of Node objects representing the graph's topological sort.
        """
        return self.tsort

    #TODO: Implement method to check that all producers' results are being read by a consumer.