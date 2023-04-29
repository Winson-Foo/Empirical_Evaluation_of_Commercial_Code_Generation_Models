from videoflow.core import Flow
from videoflow.producers import IntProducer
from videoflow.processors import IdentityProcessor, JoinerProcessor
from videoflow.consumers import CommandlineConsumer
from videoflow.core.constants import BATCH

# Constants
START = 0
STOP = 40
STEP = 0.1
FPS1 = 4
FPS2 = 2
TASKS1 = 5
TASKS2 = 10
TASKS_JOINED = 5

# Create the flow
producer = IntProducer(start=START, stop=STOP, step=STEP)
identity1 = IdentityProcessor(fps=FPS1, nb_tasks=TASKS1, name='identity1')(producer)
identity2 = IdentityProcessor(fps=FPS2, nb_tasks=TASKS2, name='identity2')(identity1)
joined = JoinerProcessor(nb_tasks=TASKS_JOINED)(identity1, identity2)
printer = CommandlineConsumer()(joined)
flow = Flow([producer], [printer], flow_type=BATCH)

# Run the flow
flow.run()
flow.join()