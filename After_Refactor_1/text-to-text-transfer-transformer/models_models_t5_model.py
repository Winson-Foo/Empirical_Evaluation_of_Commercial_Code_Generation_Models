from abc import ABC, abstractmethod

class T5Model(ABC):
  """Abstract Base class for T5 Model API."""

  @abstractmethod
  def train(self, mixture_or_task_name, steps):
    pass

  @abstractmethod
  def eval(self, mixture_or_task_name):
    pass

  @abstractmethod
  def predict(self):
    pass

  @abstractmethod
  def finetune(self, mixture_or_task_name, finetune_steps):
    pass