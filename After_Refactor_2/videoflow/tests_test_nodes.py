import pytest

from videoflow.core.graph import GraphEngine
from videoflow.core.node import Node, TaskModuleNode
from videoflow.producers import IntProducer
from videoflow.processors import IdentityProcessor, JoinerProcessor
from videoflow.consumers import CommandlineConsumer

def create_nodes():
    zero = IntProducer()
    a = IdentityProcessor()
    b = IdentityProcessor()
    c = IdentityProcessor()
    d = JoinerProcessor()
    e = IdentityProcessor()
    f = JoinerProcessor()

    return zero, a, b, c, d, e, f

def connect_nodes(zero, a, b, c, d, e, f):
    a.set_inputs(zero)
    b.set_inputs(a)
    c.set_inputs(b)
    d.set_inputs(b, c)
    e.set_inputs(d)
    f.set_inputs(d, e, c, b)

    return f

def test_taskmodule_node_simple():
    '''
    Tests simple task module creation
    and tests that it can be part of a flow
    '''
    zero, a, b, c, d, e, f = create_nodes()
    module = TaskModuleNode(a, f)
    out = CommandlineConsumer()(module)

    assert Node.count() == 8

    graph_engine = GraphEngine([zero], [out])
    tsort = graph_engine.topological_sort()

    assert len(tsort) == 3

def test_taskmodule_node_create_parents():
    '''
    Tests that task module can create its own parents without
    having to take them from the entry node.
    '''
    zero, a, b, c, d, e, f = create_nodes()
    task_module = TaskModuleNode(a, c)(zero)
    out = CommandlineConsumer()(task_module)

    assert Node.count() == 8

    graph_engine = GraphEngine([zero], [out])
    tsort = graph_engine.topological_sort()

    assert len(tsort) == 3

def test_taskmodule_node_take_childs():
    '''
    Tests that task module can take the childs from its exit_entry
    '''
    zero, a, b, c, d, e, f = create_nodes()
    out = CommandlineConsumer()(c)
    task_module = TaskModuleNode(a, c)

    assert Node.count() == 7

    graph_engine = GraphEngine([zero], [out])
    tsort = graph_engine.topological_sort()

    assert len(tsort) == 3
    assert task_module in tsort

def test_taskmodule_node_module_inside_module():
    '''
    Test error when trying to put module inside of moduel
    '''
    zero, a, b, c, d, e, f = create_nodes()
    module1 = TaskModuleNode(a, b)
    with pytest.raises(ValueError):
        module2 = TaskModuleNode(module1, c)

def test_process_algorithm():
    '''
    Test the process algorithm of the taskmodule node
    '''
    pass

if __name__ == "__main__":
    pytest.main([__file__])