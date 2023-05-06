import torch.multiprocessing as mp
from typing import List
from .agents import BaseAgent


class ActorMessage:
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5


class BaseActor(mp.Process):
    def __init__(self, config: Dict[str, Any], pipe: mp.Pipe) -> None:
        mp.Process.__init__(self)
        self.config = config
        self.pipe = pipe
        self.task = None
        self.network = None
        self.total_steps = 0
        self.cache_len = 2

    def _transition(self) -> List:
        raise NotImplementedError

    def _set_up(self) -> None:
        pass

    def _sample(self) -> List:
        transitions = []
        for _ in range(self.config.sgd_update_frequency):
            transition = self._transition()
            if transition is not None:
                transitions.append(transition)
        return transitions

    def run(self) -> None:
        self._set_up()
        self.task = self.config.task_fn()

        cache = deque([], maxlen=self.cache_len)
        while True:
            op, data = self.pipe.recv()
            if op == ActorMessage.STEP:
                if not len(cache):
                    cache.append(self._sample())
                    cache.append(self._sample())
                self.pipe.send(cache.popleft())
                cache.append(self._sample())
            elif op == ActorMessage.EXIT:
                self.pipe.close()
                return
            elif op == ActorMessage.NETWORK:
                self.network = data
            else:
                raise NotImplementedError

    def step(self) -> List:
        self.pipe.send([ActorMessage.STEP, None])
        return self.pipe.recv()

    def close(self) -> None:
        self.pipe.send([ActorMessage.EXIT, None])
        self.pipe.close()

    def set_network(self, net: torch.nn.Module) -> None:
        self.network = net