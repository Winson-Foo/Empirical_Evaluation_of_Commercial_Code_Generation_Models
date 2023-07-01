"""Internal module for attacks that support logging to TensorBoard"""

from typing import Union, Callable

class TensorBoard:
    """A custom TensorBoard class that accepts EagerPy tensors and that
    can be disabled by turned into a noop by passing logdir=False.

    This makes it possible to add tensorboard logging without any if
    statements and without any computational overhead if it's disabled.
    """

    def __init__(self, logdir: Union[bool, None, str]):
        if logdir or (logdir is None):
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(logdir=logdir)
        else:
            self.writer = None

    def noop_if_writer_None(self, f: Callable) -> Callable:
        def wrapper(self, *args, **kwds):
            if self.writer is None:
                return
            return f(self, *args, **kwds)

        return wrapper

    @noop_if_writer_None
    def close(self):
        self.writer.close()

    @noop_if_writer_None
    def scalar(self, tag: str, x: Union[int, float], step: int):
        self.writer.add_scalar(tag, x, step)

    @noop_if_writer_None
    def mean(self, tag: str, x, step: int):
        self.writer.add_scalar(tag, x.mean(axis=0).item(), step)

    @noop_if_writer_None
    def probability(self, tag: str, x, step: int):
        self.writer.add_scalar(tag, x.float32().mean(axis=0).item(), step)

    @noop_if_writer_None
    def conditional_mean(self, tag: str, x, cond, step: int):
        cond_ = cond.numpy()
        if ~cond_.any():
            return
        x_ = x.numpy()
        x_ = x_[cond_]
        self.writer.add_scalar(tag, x_.mean(axis=0).item(), step)

    @noop_if_writer_None
    def probability_ratio(self, tag: str, x, y, step: int):
        x_ = x.float32().mean(axis=0).item()
        y_ = y.float32().mean(axis=0).item()
        if y_ == 0:
            return
        self.writer.add_scalar(tag, x_ / y_, step)

    @noop_if_writer_None
    def histogram(self, tag: str, x, step: int, *, first: bool = True):
        x = x.numpy()
        self.writer.add_histogram(tag, x, step)
        if first:
            self.writer.add_scalar(tag + "/0", x[0].item(), step)