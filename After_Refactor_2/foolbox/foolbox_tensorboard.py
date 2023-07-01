from typing import Union, Callable
import eagerpy as ep
from tensorboardX import SummaryWriter
from functools import wraps


def maybenoop(f: Callable[..., None]) -> Callable[..., None]:
    @wraps(f)
    def wrapper(self: "TensorBoard", *args, **kwds) -> None:
        if self.writer is None:
            return
        return f(self, *args, **kwds)

    return wrapper


class TensorBoard:
    """A custom TensorBoard class that accepts EagerPy tensors and can be
    disabled by turning it into a noop by passing logdir=False.
    This makes it possible to add tensorboard logging without any if
    statements and without any computational overhead if it's disabled.
    """

    def __init__(self, logdir: Union[bool, str]) -> None:
        if logdir:
            self.writer = SummaryWriter(logdir=logdir)
        else:
            self.writer = None

    @maybenoop
    def close(self) -> None:
        self.writer.close()

    @maybenoop
    def scalar(self, tag: str, x: Union[int, float], step: int) -> None:
        self.writer.add_scalar(tag, x, step)

    @maybenoop
    def mean(self, tag: str, x: ep.Tensor, step: int) -> None:
        self.writer.add_scalar(tag, x.mean(axis=0).item(), step)

    @maybenoop
    def probability(self, tag: str, x: ep.Tensor, step: int) -> None:
        self.writer.add_scalar(tag, x.float32().mean(axis=0).item(), step)

    @maybenoop
    def conditional_mean(self, tag: str, x: ep.Tensor, cond: ep.Tensor, step: int) -> None:
        cond_ = cond.numpy()
        if ~cond_.any():
            return
        x_ = x.numpy()
        x_ = x_[cond_]
        self.writer.add_scalar(tag, x_.mean(axis=0).item(), step)

    @maybenoop
    def probability_ratio(self, tag: str, x: ep.Tensor, y: ep.Tensor, step: int) -> None:
        x_ = x.float32().mean(axis=0).item()
        y_ = y.float32().mean(axis=0).item()
        if y_ == 0:
            return
        self.writer.add_scalar(tag, x_ / y_, step)

    @maybenoop
    def histogram(self, tag: str, x: ep.Tensor, step: int, *, first: bool = True) -> None:
        x = x.numpy()
        self.writer.add_histogram(tag, x, step)
        if first:
            self.writer.add_scalar(tag + "/0", x[0].item(), step)