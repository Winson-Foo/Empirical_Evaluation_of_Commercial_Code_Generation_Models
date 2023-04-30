from typing import List, Tuple
import logging
import os
from multiprocessing import Process, Queue, Event, Lock

from ..core.task import Task

class TaskExecuter:
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.env_vars = {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "-1"
        }
    
    def execute_task(self, task : Task) -> None:
        # set environment variables
        self._set_env_vars()
        task.run()
    
    def execute_task_on_gpu(self, task : Task, gpu_id: int) -> None:
        # set environment variables
        self.env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self._set_env_vars()
        task.run()

    def _set_env_vars(self) -> None:
        # set environment variables
        for env_var, value in self.env_vars.items():
            os.environ[env_var] = value
        
class TaskExecutorFactory:
    
    @staticmethod
    def create_process_task(task : Task) -> Process:
        executer = TaskExecuter()
        return Process(target = executer.execute_task, args = (task,))
    
    @staticmethod
    def create_process_task_gpu(task : Task, gpu_id : int) -> Process:
        executer = TaskExecuter()
        return Process(target = executer.execute_task_on_gpu, args = (task, gpu_id)) 

# example usage
if __name__ == "__main__":
    task = Task()
    # using CPU
    p1 = TaskExecutorFactory.create_process_task(task)
    p1.start()

    # using GPU 0
    p2 = TaskExecutorFactory.create_process_task_gpu(task, 0)
    p2.start()    
    
    p1.join()
    p2.join()