import logging

from .constants import LOGGING_LEVEL
from .flow import Flow
from .node import (
    ConsumerNode,
    FunctionProcessorNode,
    ProcessorNode,
    ProducerNode,
)
from .task import ConsumerTask, ProcessorTask, ProducerTask, Task

# Configure logger
def configure_logger():
    logger = logging.getLogger(__package__)
    logger.setLevel(LOGGING_LEVEL)
    ch = logging.StreamHandler()
    ch.setLevel(LOGGING_LEVEL)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


configure_logger()

# Task classes
class Task(Task):
    pass


class ConsumerTask(ConsumerTask):
    pass


class ProcessorTask(ProcessorTask):
    pass


class ProducerTask(ProducerTask):
    pass


# Node classes
class Node(Node):
    pass


class ConsumerNode(ConsumerNode):
    pass


class ProcessorNode(ProcessorNode):
    pass


class ProducerNode(ProducerNode):
    pass


class FunctionProcessorNode(FunctionProcessorNode):
    pass


# Flow class
class Flow(Flow):
    pass