from videoflow.core import Flow
from videoflow.producers import IntProducer
from videoflow.processors import IdentityProcessor, JoinerProcessor
from videoflow.consumers import CommandlineConsumer
from videoflow.core.constants import BATCH

# Configuration variables
producer_config = {'start': 0, 'stop': 40, 'step': 0.1}
identity_config = {'fps': 4, 'nb_tasks': 5}
identity1_config = {'fps': 2, 'nb_tasks': 10}
joiner_config = {'nb_tasks': 5}

# Create components
int_producer = IntProducer(**producer_config)
identity_processor = IdentityProcessor(name='i1', **identity_config)(int_producer)
identity1_processor = IdentityProcessor(name='i2', **identity1_config)(identity_processor)
joiner_processor = JoinerProcessor(**joiner_config)(identity_processor, identity1_processor)
commandline_consumer = CommandlineConsumer()(joiner_processor)

# Create flow
flow_components = [int_producer]
flow_consumers = [commandline_consumer]
flow = Flow(flow_components, flow_consumers, flow_type=BATCH)

# Run flow
flow.run()
flow.join()