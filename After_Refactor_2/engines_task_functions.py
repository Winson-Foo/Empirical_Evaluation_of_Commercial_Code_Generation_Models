import logging
import os
from multiprocessing import Process

from ..core.task import Task


def task_executor_fn(task: Task) -> None:
    """
    Executes a given task using CPU.

    Parameters:
    task (Task): The task to be executed.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    task.run()


def task_executor_gpu_fn(task: Task, gpu_id: int) -> None:
    """
    Executes a given task using GPU.

    Parameters:
    task (Task): The task to be executed.
    gpu_id (int): The id of the GPU to use.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    task.run()


def create_process_task(task: Task) -> Process:
    """
    Creates a process for executing a task on CPU.

    Parameters:
    task (Task): The task to be executed.

    Returns:
    (Process): The created process.
    """
    return Process(target=task_executor_fn, args=(task,))


def create_process_task_gpu(task: Task, gpu_id: int) -> Process:
    """
    Creates a process for executing a task on GPU.

    Parameters:
    task (Task): The task to be executed.
    gpu_id (int): The id of the GPU to use.

    Returns:
    (Process): The created process.
    """
    return Process(target=task_executor_gpu_fn, args=(task, gpu_id))