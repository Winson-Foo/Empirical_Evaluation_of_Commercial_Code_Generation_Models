import time
from videoflow.core import Flow
from videoflow.producers import IntProducer
from videoflow.processors import IdentityProcessor, JoinerProcessor
from videoflow.consumers import CommandlineConsumer

# Constants
START = 0
STOP = 40
DELAY = 0.1
NB_TASKS = 5

def create_flow():
    producer = IntProducer(START, STOP, DELAY)
    identity = IdentityProcessor(nb_tasks=NB_TASKS)(producer)
    identity1 = IdentityProcessor(nb_tasks=NB_TASKS)(identity)
    joined = JoinerProcessor(nb_tasks=NB_TASKS)(identity, identity1)
    printer = CommandlineConsumer()(joined)
    return Flow([producer], [printer])

if __name__ == '__main__':
    flow = create_flow()
    flow.run()
    time.sleep(2)
    flow.stop()