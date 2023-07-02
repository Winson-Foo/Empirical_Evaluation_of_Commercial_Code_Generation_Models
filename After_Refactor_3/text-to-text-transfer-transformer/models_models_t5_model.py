import abc

class T5Model(metaclass=abc.ABCMeta):
    """Abstract Base class for T5 Model API."""

    @abc.abstractmethod
    def train(self, mixture_or_task_name, steps):
        """Train the T5 model on the given mixture or task for the specified number of steps."""
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluate(self, mixture_or_task_name):
        """Evaluate the T5 model on the given mixture or task."""
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self):
        """Use the T5 model to make predictions."""
        raise NotImplementedError()

    @abc.abstractmethod
    def finetune(self, mixture_or_task_name, finetune_steps):
        """Finetune the T5 model on the given mixture or task for the specified number of steps."""
        raise NotImplementedError()