from typing import Union
import pytest
import eagerpy as ep
import foolbox as fbn

@pytest.mark.parametrize("logdir", [False, "temp"])
def test_tensorboard(
    logdir: Union[False, str], tmp_path: Any, dummy: ep.Tensor
) -> None:
    if logdir == "temp":
        logdir = tmp_path

    if logdir:
        before = len(list(tmp_path.iterdir()))

    tb = fbn.tensorboard.TensorBoard(logdir)

    tb.scalar("a_scalar", 5, step=1)

    def add_tensorboard_entries(name, tensor, step):
        tb.mean(name, tensor, step=step)
        tb.probability(name, tensor, step=step)

    x = ep.ones(dummy, 10)
    add_tensorboard_entries("a_mean_a_probability", x, step=2)

    x = ep.arange(dummy, 10).float32()
    cond = ep.ones(dummy, 10) == (ep.arange(dummy, 10) % 2)
    tb.conditional_mean("a_conditional_mean", x, cond, step=2)

    cond = ep.ones(dummy, 10) == ep.zeros(dummy, 10)
    tb.conditional_mean("a_conditional_mean_false", x, cond, step=2)

    x = ep.ones(dummy, 10) == ep.arange(dummy, 10)
    y = ep.ones(dummy, 10) == (ep.arange(dummy, 10) % 2)
    tb.probability_ratio("a_probability_ratio", x, y, step=5)

    y = ep.ones(dummy, 10) == ep.zeros(dummy, 10)
    tb.probability_ratio("a_probability_ratio_y_zero", x, y, step=5)

    tb.histogram("a_histogram", x, step=9, first=False)
    tb.histogram("a_histogram", x, step=10, first=True)

    tb.close()

    if logdir:
        after = len(list(tmp_path.iterdir()))
        assert after > before  # make sure something has been written