import os
import pickle
import datetime
import time
from pathlib import Path


def get_time_str():
    """
    Generates a timestamp string.

    :return: Timestamp string.
    """
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_default_log_dir(name):
    """
    Generates the default log directory for a run.

    :param name: Name of the run.
    :return: Log directory path.
    """
    return os.path.join('log', f'{name}-{get_time_str()}')


def mkdir(path):
    """
    Creates a directory at the given path if it doesn't exist.

    :param path: Directory path.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def close_obj(obj):
    """
    Closes an object if it has a 'close' method.

    :param obj: Object to close.
    """
    if hasattr(obj, 'close'):
        obj.close()


def random_sample(indices, batch_size):
    """
    Randomly samples batches of the given size from an array of indices.

    :param indices: Array of indices to sample from.
    :param batch_size: Batch size.
    :return: Iterator over batches of indices.
    """
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


def is_plain_type(x):
    """
    Checks if an object is a primitive data type.

    :param x: Object to check.
    :return: Whether the object is a primitive data type.
    """
    for t in [str, int, float, bool]:
        if isinstance(x, t):
            return True
    return False


def generate_tag(params):
    """
    Generates a tag for a set of hyperparameters.

    :param params: Dictionary of hyperparameters.
    :return: Tag string.
    """
    if 'tag' in params.keys():
        return
    game = params['game']
    params.setdefault('run', 0)
    run = params['run']
    del params['game']
    del params['run']
    str = ['%s_%s' % (k, v if is_plain_type(v) else v.__name__) for k, v in sorted(params.items())]
    tag = '%s-%s-run-%d' % (game, '-'.join(str), run)
    params['tag'] = tag
    params['game'] = game
    params['run'] = run


def translate(pattern):
    """
    Translates a period-separated name pattern into a regex pattern.

    :param pattern: Name pattern.
    :return: Regex pattern.
    """
    groups = pattern.split('.')
    pattern = ('\.').join(groups)
    return pattern


def split(a, n):
    """
    Splits an array into approximately equal-sized chunks.

    :param a: Array to split.
    :param n: Number of chunks.
    :return: Iterator over array chunks.
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))