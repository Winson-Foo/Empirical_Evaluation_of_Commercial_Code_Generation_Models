import pytest

from videoflow.utils.graph import has_cycle, topological_sort
from videoflow.producers import IntProducer
from videoflow.processors import IdentityProcessor, JoinerProcessor

class TestGraphUtils:
    @pytest.fixture
    def simple_linear_graph(self):
        b = IdentityProcessor()
        c = IdentityProcessor()(b)
        d = IdentityProcessor()(c)
        b(c)
        return [b]

    @pytest.fixture
    def non_linear_graph(self):
        a1 = IntProducer()
        b1 = IdentityProcessor()(a1)
        c1 = IdentityProcessor()(b1)
        d1 = IdentityProcessor()(a1)
        e1 = IdentityProcessor()
        f1 = JoinerProcessor()(e1, d1)
        g1 = JoinerProcessor()(c1, b1, d1)
        e1(g1)

        a2 = IntProducer()
        b2 = IdentityProcessor()(a2)
        c2 = IdentityProcessor()(b2)
        d2 = IdentityProcessor()(a2)
        e2 = IdentityProcessor()
        f2 = JoinerProcessor()(e2, d2)
        g2 = JoinerProcessor()(c2, b2, f2)
        e2(g2)

        return [e1, e2, a1, a2]

    def test_topological_sort(self):
        a = IntProducer()
        b = IdentityProcessor()(a)
        c = IdentityProcessor()(b)
        d = IdentityProcessor()(c)
        e = IdentityProcessor()(d)

        expected_tsort = [a, b, c, d, e]
        tsort = topological_sort([a])
        assert tsort == expected_tsort, "wrong topological sort"

    def test_setting_parents_twice(self):
        b = IdentityProcessor()
        c = IdentityProcessor()(b)

        with pytest.raises(RuntimeError):
            c(b)

        with pytest.raises(RuntimeError):
            c(b)

    def test_has_cycle_on_simple_linear_graph(self, simple_linear_graph):
        assert has_cycle(simple_linear_graph), "cycle not detected on simple linear graph"

    def test_has_cycle_on_non_linear_graph(self, non_linear_graph):
        has_cycle_e1 = has_cycle([non_linear_graph[0]])
        has_cycle_e2 = has_cycle([non_linear_graph[1]])
        has_cycle_a1 = has_cycle([non_linear_graph[2]])
        has_cycle_a2 = has_cycle([non_linear_graph[3]])

        assert not has_cycle_e1, "cycle detected on non-linear graph #1"
        assert has_cycle_e2, "cycle not detected on non-linear graph #2"
        assert not has_cycle_a1, "cycle not detected on non-linear graph #3"
        assert has_cycle_a2, "cycle not detected on non-linear graph #4"

    def test_topological_sort_on_non_linear_graph(self, non_linear_graph):
        expected_tsort = [
            non_linear_graph[2], non_linear_graph[0],
            non_linear_graph[1], non_linear_graph[3]
        ]
        tsort = topological_sort(non_linear_graph)
        assert tsort == expected_tsort, "wrong topological sort on non-linear graph"

if __name__ == "__main__":
    pytest.main([__file__])