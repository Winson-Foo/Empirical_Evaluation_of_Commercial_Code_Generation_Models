from videoflow.core import Flow
from videoflow.producers import IntProducer
from videoflow.processors import IdentityProcessor, JoinerProcessor
from videoflow.consumers import CommandlineConsumer
from videoflow.core.constants import BATCH

# Create the IntProducer
reader = IntProducer(0, 100, 0.001)

# Define the processors
game_state_processor = IdentityProcessor(fps = 6, nb_tasks = 1, name = 'i1')(reader)
hero_processors = JoinerProcessor()(reader, game_state_processor)
ability_processor = JoinerProcessor()(reader, game_state_processor, hero_processors)
ammo_processor = JoinerProcessor()(reader, game_state_processor)
death_processor = JoinerProcessor()(reader, game_state_processor)
hp_processor = JoinerProcessor()(reader, game_state_processor)
killfeed_processor = JoinerProcessor(fps = 1, nb_tasks = 5)(reader, game_state_processor)
map_processor = JoinerProcessor()(reader, game_state_processor)
resurrect_processor = JoinerProcessor()(reader, game_state_processor)
sr_processor = JoinerProcessor()(reader, game_state_processor)
ultimate_processor = JoinerProcessor()(reader, game_state_processor)
player_score_processor = JoinerProcessor()(reader, game_state_processor)

# Create a list of processors for the consumer
processors_for_consumer = [
    reader, 
    game_state_processor,
    hero_processors,
    death_processor,
    killfeed_processor,
    ammo_processor,
    hp_processor,
    ultimate_processor,
    ability_processor,
    player_score_processor,
    map_processor,
    sr_processor,
    resurrect_processor
]
# Join the processors before passing to the consumer
consumer_before = JoinerProcessor()(*processors_for_consumer)

# Create the consumer
consumer = CommandlineConsumer()(consumer_before)

# Create the Flow object and run it
flow = Flow([reader], [consumer], flow_type = BATCH)
flow.run()
flow.join()
