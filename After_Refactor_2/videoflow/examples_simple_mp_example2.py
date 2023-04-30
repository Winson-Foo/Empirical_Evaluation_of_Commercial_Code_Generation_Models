from videoflow.core import Flow
from videoflow.producers import IntProducer
from videoflow.processors import IdentityProcessor, JoinerProcessor
from videoflow.consumers import CommandlineConsumer
from videoflow.core.constants import BATCH

# Create the producer
producer = IntProducer(0, 100, 0.001)

# Create the processors with descriptive names
game_state = IdentityProcessor(fps = 6, nb_tasks = 1, name = 'game_state')(producer)
hero = JoinerProcessor()(producer, game_state)
ability = JoinerProcessor()(producer, game_state, hero)
ammo = JoinerProcessor()(producer, game_state)
death = JoinerProcessor()(producer, game_state)
hp = JoinerProcessor()(producer, game_state)
killfeed = JoinerProcessor(fps = 1, nb_tasks = 5, name = 'killfeed')(producer, game_state)
map = JoinerProcessor()(producer, game_state)
resurrect = JoinerProcessor()(producer, game_state)
sr = JoinerProcessor()(producer, game_state)
ultimate = JoinerProcessor()(producer, game_state)
player_score = JoinerProcessor()(producer, game_state)

# Store the processors in a list
processors = [producer, game_state, hero, ability, ammo, death, hp,
             killfeed, map, resurrect, sr, ultimate, player_score]

# Use a dictionary to pass the inputs to the processors
inputs = {
    'producer': producer,
    'game_state': game_state,
    'hero': hero,
    'ability': ability,
    'ammo': ammo,
    'death': death,
    'hp': hp,
    'killfeed': killfeed,
    'map': map,
    'resurrect': resurrect,
    'sr': sr,
    'ultimate': ultimate,
    'player_score': player_score
}

# Use the *args syntax to pass multiple inputs to the joiner processor
consumer_before = JoinerProcessor()(*list(inputs.values()))

# Create the consumer
consumer = CommandlineConsumer()(consumer_before)

# Create the flow and run it
flow = Flow(processors, [consumer], flow_type = BATCH)
flow.run()
flow.join()