import subprocess
import os
from typing import List, Set

def get_number_of_gpus() -> int:
    '''
    Returns the number of GPUs in the system
    '''
    try:
        output = str(subprocess.check_output(["nvidia-smi", "-L"]))
        return output.count('UUID')
    except FileNotFoundError:
        return 0

def get_system_gpus() -> Set[int]:
    '''
    Returns the ids of GPUs in the machine as a set of integers
    '''
    n_gpus = get_number_of_gpus()
    return set(range(n_gpus))

def get_gpus_available_to_process() -> List[int]:
    '''
    Returns the list of ids of the GPUs available to the process calling the function.
    It first gets the set of ids of the GPUs in the system. Then it gets the set of ids marked as
    available by ``CUDA_VISIBLE_DEVICES``. It returns the intersection of those
    two sets as a list.
    '''
    system_gpus = get_system_gpus()
    visible_gpus = get_visible_gpus()
    available_gpus = system_gpus & visible_gpus
    return list(available_gpus)

def get_visible_gpus() -> Set[int]:
    '''
    Returns the set of GPUs that are visible to the process as determined by the
    CUDA_VISIBLE_DEVICES environment variable.
    '''
    env_var = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if env_var is None or env_var.strip() == '':
        return get_system_gpus()

    visible_gpus = set()
    for device in env_var.split(','):
        try:
            device_id = int(device)
            visible_gpus.add(device_id)
        except ValueError:
            pass
    return visible_gpus
