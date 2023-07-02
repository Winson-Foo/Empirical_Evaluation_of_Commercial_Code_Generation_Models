import logging
import numpy as np

class VolParams(object):
    def __init__(self, params_dict={}):
        self.data = {}
        self.volspike = {}
        self.motion = {}
        self.change_params(params_dict)

    def set_group_params(self, group, params_dict, set_if_not_exists=False, verbose=False):
        if not hasattr(self, group):
            raise KeyError(f"No group in VolParams named {group}")

        group_params = getattr(self, group)
        for key, value in params_dict.items():
            if key not in group_params and not set_if_not_exists:
                if verbose:
                    logging.warning(f"NOT setting value of key {key} in group {group}, because no prior key existed...")
            else:
                if group_params[key] != value:
                    logging.warning(f"Changing key {key} in group {group} from {group_params[key]} to {value}")
                group_params[key] = value

    def get_group_params(self, group):
        if not hasattr(self, group):
            raise KeyError(f"No group in VolParams named {group}")

        return getattr(self, group)

    def change_params(self, params_dict, verbose=False):
        for group, params in params_dict.items():
            self.set_group_params(group, params, verbose=verbose)
        for key, value in params_dict.items():
            flag = True
            for group in list(self.__dict__.keys()):
                group_params = getattr(self, group)
                if key in group_params:
                    flag = False
            if flag:
                logging.warning(f"No parameter {key} found!")

    def get(self, group, key):
        group_params = self.get_group_params(group)
        if key not in group_params:
            raise KeyError(f"No key {key} in group {group}")
        return group_params[key]