import abc


class T5Model(abc.ABC):
    """Abstract Base class for T5 Model API."""

    @abc.abstractmethod
    def train(self, mixture_or_task_name, steps):
        """Train the T5 model.

        Args:
            mixture_or_task_name: The name of the mixture or task to train on.
            steps: The number of training steps to perform.

        Raises:
            NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def eval(self, mixture_or_task_name):
        """Evaluate the T5 model.

        Args:
            mixture_or_task_name: The name of the mixture or task to evaluate.

        Raises:
            NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self):
        """Use the T5 model for prediction.

        Raises:
            NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def finetune(self, mixture_or_task_name, finetune_steps):
        """Finetune the T5 model on a specific mixture or task.

        Args:
            mixture_or_task_name: The name of the mixture or task to finetune on.
            finetune_steps: The number of finetuning steps to perform.

        Raises:
            NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError()