import os
import subprocess
from typing import List, Set


def get_number_of_gpus() -> int:
    '''
    Returns the number of GPUs in the system using nvidia-smi command.
    '''
    try:
        gpu_info = str(subprocess.check_output(['nvidia-smi', '-L']))
        return gpu_info.count('UUID')
    except FileNotFoundError:
        return 0


def get_all_gpus_in_system() -> Set[int]:
    '''
    Returns the set of IDs of GPUs in the machine as integers.
    '''
    return set(range(get_number_of_gpus()))


def get_gpus_available_to_process() -> List[int]:
    '''
    Returns the list of IDs of GPUs available to the process calling the function.
    It first gets the set of IDs of GPUs in the system. Then it gets the set of IDs marked as
    available by `CUDA_VISIBLE_DEVICES` environment variable. It returns the intersection of these
    two sets as a list.
    '''
    all_gpus = get_all_gpus_in_system()
    visible_gpus = [int(gpu_id) for gpu_id in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if gpu_id]
    available_gpus = list(all_gpus & set(visible_gpus))
    return available_gpus
