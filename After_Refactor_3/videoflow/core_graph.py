from typing import List
import logging

from ..utils.graph import has_cycle, topological_sort
from .node import ProducerNode

logger = logging.getLogger(__name__)


class GraphEngine:
    def __init__(self, producers: List[ProducerNode], consumers: List) -> None:
        self._producers = producers
        self._consumers = consumers

        self._validate_inputs()
        self._validate_graph()

        self._tsort = topological_sort(self._producers)

    def topological_sort(self) -> List:
        return list(self._tsort)

    def _validate_inputs(self) -> None:
        if len(self._producers) != 1:
            raise AttributeError('Only support flows with 1 producer for now.')

        for producer in self._producers:
            if not isinstance(producer, ProducerNode):
                raise AttributeError('{} is not instance of ProducerNode'.format(producer))

    def _validate_graph(self) -> None:
        if has_cycle(self._producers):
            logger.error('Cycle detected in computation graph. Exiting now...')
            raise ValueError('Cycle found in graph')

        for consumer in self._consumers:
            if consumer not in self._tsort:
                logger.error(f'Consumer {consumer} is not descendant of any producer. Exiting now...')
                raise ValueError(f'{consumer} is not descendant of any producer')

        # TODO: Check that all producers' results are being read by a consumer.