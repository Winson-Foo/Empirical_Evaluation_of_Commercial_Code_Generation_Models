import pytest

from videoflow.core.graph import GraphEngine
from videoflow.producers import IntProducer
from videoflow.processors import IdentityProcessor, JoinerProcessor
from videoflow.consumers import CommandlineConsumer

def setup_graph(a, b, c, d):
    return GraphEngine([a], [d])

def test_no_raise_error():
    producer = IntProducer()
    identity1 = IdentityProcessor()
    identity2 = IdentityProcessor()
    consumer = CommandlineConsumer()

    b = identity1(producer)
    c = identity2(b)

    graph_engine = setup_graph(producer, b, c, consumer)

def test_raise_error_missing_consumer():
    producer = IntProducer()
    identity1 = IdentityProcessor()
    identity2 = IdentityProcessor()

    b = identity1(producer)
    c = identity2(b)

    with pytest.raises(ValueError):
        graph_engine = setup_graph(producer, b, c, CommandlineConsumer())

def test_raise_error_missing_processor():
    producer = IntProducer()
    identity1 = IdentityProcessor()
    identity2 = IdentityProcessor()
    consumer = CommandlineConsumer()

    b = identity1(producer)
    d = identity2()
    e = consumer(d)

    with pytest.raises(ValueError):
        graph_engine = setup_graph(producer, b, e)

if __name__ == "__main__":
    pytest.main([__file__])