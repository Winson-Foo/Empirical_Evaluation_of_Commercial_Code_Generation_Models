from videoflow.core import Flow
from videoflow.core.node import TaskModuleNode
from videoflow.consumers import CommandlineConsumer
from videoflow.processors import IdentityProcessor, JoinerProcessor
from videoflow.producers import IntProducer

def get_identity_processor(producer: IntProducer):
    return IdentityProcessor(nb_tasks=1)(producer)

def get_joiner_processor(identity_0: IdentityProcessor, identity_1: IdentityProcessor):
    return JoinerProcessor(nb_tasks=1)(identity_0, identity_1)

def get_task_module_node(identity_0: IdentityProcessor, identity_1: IdentityProcessor):
    return TaskModuleNode(identity_0, identity_1)

def get_commandline_consumer(task: TaskModuleNode):
    return CommandlineConsumer()(task)

def run_flow(producer: IntProducer, consumer: CommandlineConsumer):
    flow = Flow([producer], [consumer])
    flow.run()
    flow.join()

if __name__ == '__main__':
    producer = IntProducer(start=0, end=40, sleep_time=0.05)
    identity_0 = get_identity_processor(producer)
    identity_1 = get_identity_processor(identity_0)
    joined = get_joiner_processor(identity_0, identity_1)
    task_module = get_task_module_node(identity_0, identity_1)
    printer = get_commandline_consumer(task_module)
    run_flow(producer, printer)