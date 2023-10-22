from videoflow.core import Flow
from videoflow.core.node import TaskModuleNode
from videoflow.consumers import CommandlineConsumer
from videoflow.processors import IdentityProcessor, JoinerProcessor
from videoflow.producers import IntProducer


def create_identity_processor():
    return IdentityProcessor(nb_tasks=1)


def create_flow():
    producer = IntProducer(0, 40, 0.05)
    identity = create_identity_processor()(producer)
    identity1 = create_identity_processor()(identity)

    joined = JoinerProcessor(nb_tasks=1)(identity, identity1)
    task_module = TaskModuleNode(identity, joined)
    printer = CommandlineConsumer()(task_module)

    return Flow([producer], [printer])


if __name__ == '__main__':
    flow = create_flow()
    flow.run()
    flow.join()
