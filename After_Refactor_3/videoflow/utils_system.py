import subprocess
import os
from typing import Set, List

class GPUManager:
    
    def __init__(self):
        pass
        
    def _get_number_of_gpus(self) -> int:
        '''
        Returns the number of gpus in the system
        '''
        try:
            n = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
            return n
        except FileNotFoundError:
            return 0

    def get_system_gpus(self) -> Set[int]:
        '''
        Returns the ids of gpus in the machine as a set of integers
        '''
        n = self._get_number_of_gpus()
        return set(range(n))

    def get_gpus_available_to_process(self) -> List[int]:
        '''
        Returns the list of ids of the gpus available to the process calling the function.
        It first gets the set of ids of the gpus in the system.  Then it gets the set of ids marked as
        available by ``CUDA_VISIBLE_DEVICES``. It returns the intersection of those
        two sets as a list.
        '''
        system_devices = self.get_system_gpus()
        env_var = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if env_var is None:
            visible_devices = set(system_devices)
        else:
            env_var = env_var.strip()
            visible_devices = set()
            if len(env_var) > 0:
                devices = env_var.split(',')
                for device in devices:
                    try:
                        device_id = int(device)
                        visible_devices.add(device_id)
                    except:
                        pass

        available_devices = system_devices & visible_devices
        return list(available_devices)