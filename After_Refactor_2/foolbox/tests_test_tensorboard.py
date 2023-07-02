import pytest
import eagerpy as ep
import foolbox as fbn


def setup_tensorboard(logdir, tmp_path):
    if logdir == "temp":
        logdir = tmp_path

    if logdir:
        before = len(list(tmp_path.iterdir()))

    tensorboard = fbn.tensorboard.TensorBoard(logdir)

    tensorboard.scalar("a_scalar", 5, step=1)

    x = ep.ones(dummy, 10)
    tensorboard.mean("a_mean", x, step=2)

    x = ep.ones(dummy, 10) == ep.arange(dummy, 10)
    tensorboard.probability("a_probability", x, step=2)

    x = ep.arange(dummy, 10).float32()
    cond = ep.ones(dummy, 10) == (ep.arange(dummy, 10) % 2)
    tensorboard.conditional_mean("a_conditional_mean", x, cond, step=2)

    x = ep.arange(dummy, 10).float32()
    cond = ep.ones(dummy, 10) == ep.zeros(dummy, 10)
    tensorboard.conditional_mean("a_conditional_mean_false", x, cond, step=2)

    x = ep.ones(dummy, 10) == ep.arange(dummy, 10)
    y = ep.ones(dummy, 10) == (ep.arange(dummy, 10) % 2)
    tensorboard.probability_ratio("a_probability_ratio", x, y, step=5)

    x = ep.ones(dummy, 10) == (ep.arange(dummy, 10) % 2)
    y = ep.ones(dummy, 10) == ep.zeros(dummy, 10)
    tensorboard.probability_ratio("a_probability_ratio_y_zero", x, y, step=5)

    tensorboard.close()

    if logdir:
        after = len(list(tmp_path.iterdir()))
        assert after > before  # make sure something has been written


@pytest.mark.parametrize("logdir", [False, "temp"])
def test_tensorboard(logdir, tmp_path, dummy):
    setup_tensorboard(logdir, tmp_path)