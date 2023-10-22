import pytest

from videoflow.core.graph import GraphEngine
from videoflow.producers import IntProducer
from videoflow.processors import IdentityProcessor, JoinerProcessor
from videoflow.consumers import CommandlineConsumer

def create_pipeline(producer, processors, consumer):
    pipeline = [producer]
    pipeline += processors
    pipeline.append(consumer)

    return pipeline

def test_no_raise_error():
    pipeline = create_pipeline(
        IntProducer(),
        [IdentityProcessor(), IdentityProcessor()],
        CommandlineConsumer()
    )

    graph_engine = GraphEngine(pipeline)

def test_raise_error_single_consumer():
    pipeline = create_pipeline(
        IntProducer(),
        [IdentityProcessor(), IdentityProcessor()],
        CommandlineConsumer()
    )

    with pytest.raises(ValueError):
        GraphEngine(pipeline[:-1])

def test_raise_error_invalid_connection():
    pipeline1 = create_pipeline(
        IntProducer(),
        [IdentityProcessor(), IdentityProcessor()],
        CommandlineConsumer()
    )

    pipeline2 = create_pipeline(
        IdentityProcessor(),
        [IdentityProcessor()],
        CommandlineConsumer()
    )

    with pytest.raises(ValueError):
        GraphEngine(pipeline1 + pipeline2)

if __name__ == "__main__":
    pytest.main([__file__])
