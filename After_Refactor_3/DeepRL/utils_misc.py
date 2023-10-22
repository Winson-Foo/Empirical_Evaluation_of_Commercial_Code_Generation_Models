import time
import datetime
from pathlib import Path
from typing import List
from collections import Sequence

from .agent import Agent


def run_agent(agent: Agent) -> None:
    """
    Runs the agent and logs training progress at regular intervals.

    :param agent: the RL agent to run.
    """
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save('data/%s-%s-%d' % (agent_name, config.tag, agent.total_steps))
        if config.log_interval and not agent.total_steps % config.log_interval:
            agent.logger.info('steps %d, %.2f steps/s' % (agent.total_steps, config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()
        agent.switch_task()


def get_time_str() -> str:
    """
    Returns the current date and time as a formatted string.

    :return: the current date and time as a formatted string.
    """
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_default_log_dir(name: str) -> str:
    """
    Returns the default log directory for a given logger name.

    :param name: the name of the logger.
    :return: the default log directory for the logger.
    """
    return f'./log/{name}-{get_time_str()}'


def mkdir(path: str) -> None:
    """
    Creates a new directory if it does not already exist.

    :param path: the path of the directory to be created.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def close_obj(obj) -> None:
    """
    Closes an object if it has a `close` method.

    :param obj: the object to be closed.
    """
    if hasattr(obj, 'close'):
        obj.close()


def random_sample(indices: List[int], batch_size: int) -> List[int]:
    """
    Randomly samples a batch of indices from a list of indices.

    :param indices: the list of indices to sample from.
    :param batch_size: the size of the sample batch.
    :return: a list of indices of size `batch_size`.
    """
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


def is_plain_type(x: any) -> bool:
    """
    Determines whether a given object is a plain type.

    :param x: the object to check.
    :return: True if the object is a plain type, False otherwise.
    """
    for t in [str, int, float, bool]:
        if isinstance(x, t):
            return True
    return False


def generate_tag(params: dict) -> None:
    """
    Generates a tag for the given parameter dictionary.

    :param params: a dictionary of hyperparameters.
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


def translate(pattern: str) -> str:
    """
    Translates a string pattern to a regular expression pattern.

    :param pattern: the string pattern to translate.
    :return: the corresponding regular expression pattern.
    """
    groups = pattern.split('.')
    pattern = ('\.').join(groups)
    return pattern


def split(a: List, n: int) -> List[List]:
    """
    Splits a list into `n` sublists of approximately equal size.

    :param a: the list to split.
    :param n: the number of sublists to split into.
    :return: a list of `n` sublists.
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


class HyperParameter:
    """
    A named hyperparameter value for a given key.
    """
    def __init__(self, id: int, param: dict) -> None:
        """
        Initialize a new instance of the HyperParameter class.

        :param id: the id of the hyperparameter.
        :param param: the dictionary of parameter key-value pairs.
        """
        self.id = id
        self.param = dict()
        for key, item in param:
            self.param[key] = item

    def __str__(self) -> str:
        """
        Returns the string representation of the HyperParameter object.

        :return: the string representation of the HyperParameter object.
        """
        return str(self.id)

    def dict(self) -> dict:
        """
        Returns the dictionary representation of the HyperParameter object.

        :return: the dictionary representation of the HyperParameter object.
        """
        return self.param


class HyperParameters(Sequence):
    """
    Collection of hyperparameters to be tested.
    """
    def __init__(self, ordered_params: OrderedDict) -> None:
        """
        Initialize a new instance of the HyperParameters class.

        :param ordered_params: an OrderedDict containing the parameter keys and values to be tested.
        """
        if not isinstance(ordered_params, OrderedDict):
            raise NotImplementedError
        params = []
        for key in ordered_params.keys():
            param = [[key, iterm] for iterm in ordered_params[key]]
            params.append(param)
        self.params = list(itertools.product(*params))

    def __getitem__(self, index: int) -> HyperParameter:
        """
        Returns the hyperparameter object at the given index.

        :param index: the index of the hyperparameter object to return.
        :return: the hyperparameter object at index `index`.
        """
        return HyperParameter(index, self.params[index])

    def __len__(self) -> int:
        """
        Returns the number of hyperparameter objects in the collection.

        :return: the number of hyperparameter objects in the collection.
        """
        return len(self.params)