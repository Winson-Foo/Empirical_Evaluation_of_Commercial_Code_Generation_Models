import pytest

from videoflow.utils.graph import has_cycle, topological_sort
from videoflow.producers import IntProducer
from videoflow.processors import IdentityProcessor, JoinerProcessor

# reusable functions
def create_linear_graph(n):
    prev_node = IntProducer()
    nodes = [prev_node]
    for i in range(n):
        node = IdentityProcessor()(prev_node)
        nodes.append(node)
        prev_node = node
    return nodes

def create_nonlinear_graph():
    a = IntProducer()
    b = IdentityProcessor()(a)
    c = IdentityProcessor()(b)
    d = IdentityProcessor()(a)
    e = IdentityProcessor()
    f = JoinerProcessor()(e, d)
    g = JoinerProcessor()(c, b, d)
    e(g)
    return [a, b, c, d, e, f, g]

def test_topological_sort():
    nodes = create_linear_graph(5)
    expected_tsort = nodes
    tsort = topological_sort([nodes[0]])
    assert len(tsort) == len(expected_tsort), "topological sort returned different number of nodes"
    assert all([tsort[i] is expected_tsort[i] for i in range(len(tsort))]), "wrong topological sort"

def test_setting_parents_twice():
    a = IdentityProcessor()
    b = IdentityProcessor()(a)
    
    with pytest.raises(RuntimeError):
        b(a)

def test_cycle_detection():
    # 1. simple linear graph with cycle
    nodes = create_linear_graph(3)
    nodes[-1](nodes[0])
    assert has_cycle([nodes[0]]), '#1 Cycle not detected'

    # 2. more complex nonlinear graph
    nodes = create_nonlinear_graph()
    assert not has_cycle([nodes[4]]), "#2 Cycle detected"
    assert not has_cycle([nodes[0]]), "#3 Cycle not detected"

    # 3. another nonlinear graph with cycle
    a = IntProducer()
    b = IdentityProcessor()(a)
    c = IdentityProcessor()(b)
    d = IdentityProcessor()(a)
    e = IdentityProcessor()
    f = JoinerProcessor()(e, d)
    g = JoinerProcessor()(c, b, f)
    e(g)
    assert has_cycle([e]), '#4 Cycle not detected'
    assert has_cycle([a]), "#5 Cycle not detected"

if __name__ == "__main__":
    pytest.main([__file__])