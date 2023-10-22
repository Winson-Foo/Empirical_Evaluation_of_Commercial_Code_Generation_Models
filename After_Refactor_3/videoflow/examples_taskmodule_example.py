from typing import List

from videoflow.core import Flow
from videoflow.core.node import TaskModuleNode
from videoflow.producers import IntProducer
from videoflow.processors import IdentityProcessor, JoinerProcessor
from videoflow.consumers import CommandlineConsumer


def create_producer() -> IntProducer:
    return IntProducer(start=0, stop=40, interval=0.05)


def create_identity_processor(input_node) -> IdentityProcessor:
    return IdentityProcessor(nb_tasks=1)(input_node)


def create_joiner_processor(input_nodes: List) -> JoinerProcessor:
    return JoinerProcessor(nb_tasks=1)(*input_nodes)


def main():
    producer = create_producer()
    identity = create_identity_processor(producer)
    identity1 = create_identity_processor(identity)
    joined = create_joiner_processor([identity, identity1])
    task_module = TaskModuleNode(identity, joined)
    printer = CommandlineConsumer()(task_module)
    flow = Flow([producer], [printer])
    flow.run()
    flow.join()


if __name__ == '__main__':
    main()