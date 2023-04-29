import logging
import os
from multiprocessing import Process

from ..core.task import Task

def set_cuda_environment_variables(gpu_id: int = None) -> None:
    """Sets CUDA environment variables."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def run_task(task: Task) -> None:
    """Runs a task using CPU."""
    set_cuda_environment_variables()
    task.run()

def run_task_on_gpu(task: Task, gpu_id: int) -> None:
    """Runs a task using a specific GPU."""
    set_cuda_environment_variables(gpu_id)
    task.run()

def create_process_for_task(task: Task) -> Process:
    """Creates a process to run a task using CPU."""
    return Process(target=run_task, args=(task,))

def create_process_for_task_on_gpu(task: Task, gpu_id: int) -> Process:
    """Creates a process to run a task using a specific GPU."""
    return Process(target=run_task_on_gpu, args=(task, gpu_id))