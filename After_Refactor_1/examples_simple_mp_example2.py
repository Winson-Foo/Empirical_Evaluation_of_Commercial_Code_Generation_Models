from videoflow.core import Flow
from videoflow.producers import IntProducer
from videoflow.processors import (
    IdentityProcessor, 
    JoinerProcessor,
)
from videoflow.consumers import CommandlineConsumer
from videoflow.core.constants import BATCH


def create_game_state_processor(reader):
    return IdentityProcessor(
        fps=6, 
        nb_tasks=1, 
        name='game_state_processor'
    )(reader)


def create_hero_processors(reader, game_state_processor):
    return JoinerProcessor()(reader, game_state_processor)


def create_ability_processor(reader, game_state_processor, hero_processors):
    return JoinerProcessor()(reader, game_state_processor, hero_processors)


def create_ammo_processor(reader, game_state_processor):
    return JoinerProcessor()(reader, game_state_processor)


def create_death_processor(reader, game_state_processor):
    return JoinerProcessor()(reader, game_state_processor)


def create_hp_processor(reader, game_state_processor):
    return JoinerProcessor()(reader, game_state_processor)


def create_killfeed_processor(reader, game_state_processor):
    return JoinerProcessor(
        fps=1, 
        nb_tasks=5, 
        name='killfeed_processor'
    )(reader, game_state_processor)


def create_map_processor(reader, game_state_processor):
    return JoinerProcessor()(reader, game_state_processor)


def create_resurrect_processor(reader, game_state_processor):
    return JoinerProcessor()(reader, game_state_processor)


def create_sr_processor(reader, game_state_processor):
    return JoinerProcessor()(reader, game_state_processor)


def create_ultimate_processor(reader, game_state_processor):
    return JoinerProcessor()(reader, game_state_processor)


def create_player_score_processor(reader, game_state_processor):
    return JoinerProcessor()(reader, game_state_processor)


def create_consumer(reader, game_state_processor, hero_processors, death_processor,
                    killfeed_processor, ammo_processor, hp_processor, 
                    ultimate_processor, ability_processor, player_score_processor, 
                    map_processor, sr_processor, resurrect_processor):
    processors = [
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
        resurrect_processor,
    ]
    return JoinerProcessor()(processors)


READER_START = 0
READER_STOP = 100
READER_STEP = 0.001


reader = IntProducer(start=READER_START, stop=READER_STOP, step=READER_STEP)

game_state_processor = create_game_state_processor(reader)

hero_processors = create_hero_processors(reader, game_state_processor)

ability_processor = create_ability_processor(
    reader, game_state_processor, hero_processors
)

ammo_processor = create_ammo_processor(reader, game_state_processor)

death_processor = create_death_processor(reader, game_state_processor)

hp_processor = create_hp_processor(reader, game_state_processor)

killfeed_processor = create_killfeed_processor(reader, game_state_processor)

map_processor = create_map_processor(reader, game_state_processor)

resurrect_processor = create_resurrect_processor(reader, game_state_processor)

sr_processor = create_sr_processor(reader, game_state_processor)

ultimate_processor = create_ultimate_processor(reader, game_state_processor)

player_score_processor = create_player_score_processor(
    reader, game_state_processor
)

consumer = create_consumer(
    reader, game_state_processor, hero_processors, death_processor,
    killfeed_processor, ammo_processor, hp_processor, ultimate_processor, 
    ability_processor, player_score_processor, map_processor, sr_processor, 
    resurrect_processor
)

flow = Flow([reader], [consumer], flow_type=BATCH)
flow.run()
flow.join()
