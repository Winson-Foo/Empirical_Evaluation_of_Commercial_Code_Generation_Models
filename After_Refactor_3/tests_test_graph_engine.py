import pytest
import logging

from videoflow.core.graph import GraphEngine
from videoflow.producers import IntProducer
from videoflow.processors import IdentityProcessor, JoinerProcessor
from videoflow.consumers import CommandlineConsumer

# Constants
PRODUCER_CONFIG = {}
IDENTITY_PROCESSOR_CONFIG = {}
JOINER_PROCESSOR_CONFIG = {}
COMMANDLINE_CONSUMER_CONFIG = {}

# Helper functions
def create_int_producer(config):
    return IntProducer(**config)

def create_identity_processor(config):
    return IdentityProcessor(**config)

def create_joiner_processor(config):
    return JoinerProcessor(**config)

def create_commandline_consumer(config):
    return CommandlineConsumer(**config)

# Test cases
def test_no_raise_error():
    a = create_int_producer(PRODUCER_CONFIG)
    b = create_identity_processor(IDENTITY_PROCESSOR_CONFIG)(a)
    c = create_identity_processor(IDENTITY_PROCESSOR_CONFIG)(b)
    d = create_commandline_consumer(COMMANDLINE_CONSUMER_CONFIG)(c)

    graph_engine = GraphEngine([a], [d])

def test_raise_error_1():
    a = create_int_producer(PRODUCER_CONFIG)
    b = create_identity_processor(IDENTITY_PROCESSOR_CONFIG)(a)
    c = create_identity_processor(IDENTITY_PROCESSOR_CONFIG)(b)
    d = create_commandline_consumer(COMMANDLINE_CONSUMER_CONFIG)

    with pytest.raises(ValueError):
        graph_engine = GraphEngine([a], [d])

def test_raise_error_2():
    a = create_int_producer(PRODUCER_CONFIG)
    b = create_identity_processor(IDENTITY_PROCESSOR_CONFIG)(a)
    c = create_identity_processor(IDENTITY_PROCESSOR_CONFIG)(b)
    d = create_identity_processor(IDENTITY_PROCESSOR_CONFIG)
    e = create_commandline_consumer(COMMANDLINE_CONSUMER_CONFIG)(d)

    with pytest.raises(ValueError):
        graph_engine = GraphEngine([a], [e])

# Main function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info('Starting tests.')
    pytest.main([__file__])
    logging.info('Tests finished.')