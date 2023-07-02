from typing import Union, Any
from typing_extensions import Literal
import pytest
import eagerpy as ep
import foolbox as fbn


def setup_tensorboard(logdir: Union[Literal[False], None, str], tmp_path: Any) -> fbn.tensorboard.TensorBoard:
    if logdir == "temp":
        logdir = tmp_path

    tb = fbn.tensorboard.TensorBoard(logdir)
    return tb


def log_scalar(tb: fbn.tensorboard.TensorBoard):
    tb.scalar("a_scalar", 5, step=1)


def log_mean(tb: fbn.tensorboard.TensorBoard, dummy: ep.Tensor):
    x = ep.ones(dummy, 10)
    tb.mean("a_mean", x, step=2)


def log_probability(tb: fbn.tensorboard.TensorBoard, dummy: ep.Tensor):
    x = ep.ones(dummy, 10) == ep.arange(dummy, 10)
    tb.probability("a_probability", x, step=2)


def log_conditional_mean(tb: fbn.tensorboard.TensorBoard, dummy: ep.Tensor):
    x = ep.arange(dummy, 10).float32()
    cond = ep.ones(dummy, 10) == (ep.arange(dummy, 10) % 2)
    tb.conditional_mean("a_conditional_mean", x, cond, step=2)


def log_conditional_mean_false(tb: fbn.tensorboard.TensorBoard, dummy: ep.Tensor):
    x = ep.arange(dummy, 10).float32()
    cond = ep.ones(dummy, 10) == ep.zeros(dummy, 10)
    tb.conditional_mean("a_conditional_mean_false", x, cond, step=2)


def log_probability_ratio(tb: fbn.tensorboard.TensorBoard, dummy: ep.Tensor):
    x = ep.ones(dummy, 10) == ep.arange(dummy, 10)
    y = ep.ones(dummy, 10) == (ep.arange(dummy, 10) % 2)
    tb.probability_ratio("a_probability_ratio", x, y, step=5)


def log_probability_ratio_y_zero(tb: fbn.tensorboard.TensorBoard, dummy: ep.Tensor):
    x = ep.ones(dummy, 10) == (ep.arange(dummy, 10) % 2)
    y = ep.ones(dummy, 10) == ep.zeros(dummy, 10)
    tb.probability_ratio("a_probability_ratio_y_zero", x, y, step=5)


def log_histogram(tb: fbn.tensorboard.TensorBoard, dummy: ep.Tensor):
    x = ep.arange(dummy, 10).float32()
    tb.histogram("a_histogram", x, step=9, first=False)
    tb.histogram("a_histogram", x, step=10, first=True)


def close_tensorboard(tb: fbn.tensorboard.TensorBoard):
    tb.close()


@pytest.mark.parametrize("logdir", [False, "temp"])
def test_tensorboard(logdir: Union[Literal[False], None, str], tmp_path: Any, dummy: ep.Tensor) -> None:
    if logdir:
        before = len(list(tmp_path.iterdir()))

    tb = setup_tensorboard(logdir, tmp_path)

    log_scalar(tb)
    log_mean(tb, dummy)
    log_probability(tb, dummy)
    log_conditional_mean(tb, dummy)
    log_conditional_mean_false(tb, dummy)
    log_probability_ratio(tb, dummy)
    log_probability_ratio_y_zero(tb, dummy)
    log_histogram(tb, dummy)

    close_tensorboard(tb)

    if logdir:
        after = len(list(tmp_path.iterdir()))
        assert after > before  # make sure something has been written