from videoflow.consumers import CommandlineConsumer
from videoflow.core import Flow
from videoflow.core.constants import BATCH
from videoflow.processors import IdentityProcessor, JoinerProcessor
from videoflow.producers import IntProducer

producer = IntProducer(start=0, stop=40, interval=0.1)
identity1_processor = IdentityProcessor(fps=4, nb_tasks=5, name='identity1')
identity1 = identity1_processor(producer)
identity2_processor = IdentityProcessor(fps=2, nb_tasks=10, name='identity2')
identity2 = identity2_processor(identity1)
joiner_processor = JoinerProcessor(nb_tasks=5)
joined = joiner_processor(identity1, identity2)
commandline_consumer = CommandlineConsumer()
printer = commandline_consumer(joined)
flow = Flow([producer], [printer], flow_type=BATCH)
flow.run()
flow.join()