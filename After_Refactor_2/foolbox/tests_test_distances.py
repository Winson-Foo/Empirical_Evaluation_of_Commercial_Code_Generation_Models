from typing import Dict, Callable
import foolbox as fbn
import eagerpy as ep

distances: Dict[float, Callable] = {
    0: fbn.distances.l0,
    1: fbn.distances.l1,
    2: fbn.distances.l2,
    ep.inf: fbn.distances.linf,
}