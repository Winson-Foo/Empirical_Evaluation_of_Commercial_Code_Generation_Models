import pytest

from videoflow.core.graph import GraphEngine
from videoflow.core.node import TaskModuleNode
from videoflow.producers import IntProducer
from videoflow.processors import IdentityProcessor, JoinerProcessor
from videoflow.consumers import CommandlineConsumer


def create_pipeline():
    # define the pipeline for testing
    producer = IntProducer()

    identity_processor_1 = IdentityProcessor()
    identity_processor_2 = IdentityProcessor()
    identity_processor_3 = IdentityProcessor()

    joiner_processor_1 = JoinerProcessor()
    joiner_processor_2 = JoinerProcessor()

    node_a = identity_processor_1(producer)
    node_b = identity_processor_2(node_a)
    node_c = identity_processor_3(node_b)
    node_d = joiner_processor_1(node_b, node_c)
    node_e = identity_processor_3(node_d)
    node_f = joiner_processor_2(node_d, node_e, node_c, node_b)

    task_module = TaskModuleNode(node_a, node_f)

    CommandlineConsumer()(task_module)
    graph_engine = GraphEngine([producer], [task_module])
    return graph_engine


def test_taskmodule_node():
    '''
    Tests simple task module creation
    and tests that it can be part of a flow
    '''
    graph_engine = create_pipeline()
    tsort = graph_engine.topological_sort()

    assert len(tsort) == 3


def test_taskmodule_node_1():
    '''
    Tests that task module can create its own parents without
    having to take them from the entry node.
    '''
    producer = IntProducer()
    identity_processor = IdentityProcessor()
    node_a = identity_processor(producer)
    node_b = identity_processor(node_a)
    node_c = identity_processor(node_b)
    task_module = TaskModuleNode(node_a, node_c)(producer)
    CommandlineConsumer()(task_module)
    graph_engine = GraphEngine([producer], [task_module])


def test_taskmodule_node_2():
    '''
    Tests that task module can take the childs from its exit_entry
    '''
    producer = IntProducer()
    identity_processor_1 = IdentityProcessor()
    identity_processor_2 = IdentityProcessor()
    identity_processor_3 = IdentityProcessor()

    node_a = identity_processor_1(producer)
    node_b = identity_processor_2(node_a)
    node_c = identity_processor_3(node_b)

    task_module = TaskModuleNode(node_a, node_c)

    CommandlineConsumer()(node_c)

    graph_engine = GraphEngine([producer], [node_c])
    tsort = graph_engine.topological_sort()

    assert len(tsort) == 3
    assert task_module in tsort


def test_taskmodule_node_3():
    '''
    Test error when trying to put module inside of moduel
    '''
    producer = IntProducer()

    identity_processor_1 = IdentityProcessor()
    identity_processor_2 = IdentityProcessor()
    identity_processor_3 = IdentityProcessor()

    joiner_processor_1 = JoinerProcessor()
    joiner_processor_2 = JoinerProcessor()

    node_a = identity_processor_1(producer)
    node_b = identity_processor_2(node_a)
    node_c = identity_processor_3(node_b)
    node_d = joiner_processor_1(node_b, node_c)
    node_e = identity_processor_3(node_d)
    node_f = joiner_processor_2(node_d, node_e, node_c, node_b)

    module_1 = TaskModuleNode(node_a, node_f)

    identity_processor_4 = IdentityProcessor()
    identity_processor_5 = IdentityProcessor()
    identity_processor_6 = IdentityProcessor()

    node_g = identity_processor_4(module_1)
    node_h = identity_processor_5(node_g)
    node_i = identity_processor_6(node_h)

    with pytest.raises(ValueError):
        TaskModuleNode(module_1, node_i)


def test_taskmodule_node_4():
    '''
    Test the process algorithm of the taskmodule node
    '''
    assert True


if __name__ == "__main__":
    pytest.main([__file__])