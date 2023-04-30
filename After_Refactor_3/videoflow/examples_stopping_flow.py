import time

from videoflow.core import Flow
from videoflow.producers import IntProducer
from videoflow.processors import IdentityProcessor, JoinerProcessor
from videoflow.consumers import CommandlineConsumer

# Create the producer and processors
num_tasks = 5
producer = IntProducer(start=0, stop=40, step=0.1)
identity_processor_1 = IdentityProcessor(nb_tasks=num_tasks)(producer)
identity_processor_2 = IdentityProcessor(nb_tasks=num_tasks)(identity_processor_1)
joined_processor = JoinerProcessor(nb_tasks=num_tasks)(identity_processor_1, identity_processor_2)
printer_consumer = CommandlineConsumer()(joined_processor)

# Create the flow and run it
flow = Flow([producer], [printer_consumer])
flow.run()

# Wait for a few seconds
time.sleep(2)

# Stop the flow
flow.stop()