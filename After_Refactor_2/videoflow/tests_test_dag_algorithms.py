import pytest

from videoflow.utils.graph import has_cycle, topological_sort
from videoflow.producers import IntProducer
from videoflow.processors import IdentityProcessor, JoinerProcessor

class TestVideoFlow:
    def setUp(self):
        self.a = IntProducer()
        self.b = IdentityProcessor()
        self.c = IdentityProcessor()
        self.d = IdentityProcessor()
        self.e = IdentityProcessor()
        self.f = JoinerProcessor()
        self.g = JoinerProcessor()

    def test_topological_sort(self):
        expected_tsort = [self.a, self.b, self.c, self.d, self.e]
        tsort = topological_sort([self.a])
        assert len(tsort) == len(expected_tsort), "topological sort returned different number of nodes"
        assert all([tsort[i] is expected_tsort[i] for i in range(len(tsort))]), "wrong topological sort"

    def test_setting_parents_twice(self):
        with pytest.raises(RuntimeError):
            self.c(self.b)

        with pytest.raises(RuntimeError):
            self.c(self.b)

    def test_cycle_detection(self):
        #1. simple linear graph with cycle
        self.b(self.c)
        self.c(self.d)
        self.d(self.e)
        assert has_cycle([self.b]), '#1 Cycle not detected'

        #2. More complex non linear graph
        self.f.set_processors([self.e, self.d])
        self.g.set_processors([self.c, self.b, self.d])
        self.e(self.g)
        self.assertEqual(has_cycle([self.e]), False)

        self.f.set_processors([self.e, self.d])
        self.g.set_processors([self.c, self.b, self.f])
        self.e(self.g)
        assert has_cycle([self.e]), '#4 Cycle not detected'

if __name__ == "__main__":
    pytest.main([__file__])