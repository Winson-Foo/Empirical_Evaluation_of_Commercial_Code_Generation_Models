import logging
import numpy as np

class VolParams(object):
    def __init__(self, fnames=None, fr=None, index=None, ROIs=None, weights=None,
                 template_size=0.02, context_size=35, censor_size=12, visualize_ROI=False, flip_signal=True, 
                 hp_freq_pb=1/3, nPC_bg=8, ridge_bg=0.01, hp_freq=1, clip=100, 
                 threshold_method='adaptive_threshold', min_spikes=10, pnorm=0.5, threshold=3, 
                 sigmas=np.array([1, 1.5, 2]), n_iter=2, weight_update='ridge', do_plot=False,  
                 do_cross_val=False, sub_freq=20, method='spikepursuit', superfactor=10, params_dict={}):
        """Class for setting parameters for voltage imaging. Including parameters for the data, motion correction and
        spike detection. The preferred way to set parameters is by using the set function, where a subclass is determined
        and a dictionary is passed. The whole dictionary can also be initialized at once by passing a dictionary
        params_dict when initializing the CNMFParams object.
        """
        self.data = {
            'fnames': fnames,
            'fr': fr,
            'index': index,
            'ROIs': ROIs,
            'weights': weights
        }

        self.volspike = {
            'template_size': template_size,
            'context_size': context_size,
            'censor_size': censor_size,
            'visualize_ROI': visualize_ROI,
            'flip_signal': flip_signal,
            'hp_freq_pb': hp_freq_pb,
            'nPC_bg': nPC_bg,
            'ridge_bg': ridge_bg,
            'hp_freq': hp_freq,
            'clip': clip,
            'threshold_method': threshold_method,
            'min_spikes': min_spikes,
            'pnorm': pnorm,
            'threshold': threshold,
            'sigmas': sigmas,
            'n_iter': n_iter,
            'weight_update': weight_update,
            'do_plot': do_plot,
            'do_cross_val': do_cross_val,
            'sub_freq': sub_freq,
            'method': method,
            'superfactor': superfactor
        }

        self.motion = {
            'border_nan': 'copy',
            'gSig_filt': None,
            'max_deviation_rigid': 3,
            'max_shifts': (6, 6),
            'min_mov': None,
            'niter_rig': 1,
            'nonneg_movie': True,
            'num_frames_split': 80,
            'num_splits_to_process_els': None,
            'num_splits_to_process_rig': None,
            'overlaps': (32, 32),
            'pw_rigid': False,
            'shifts_opencv': True,
            'splits_els': 14,
            'splits_rig': 14,
            'strides': (96, 96),
            'upsample_factor_grid': 4,
            'use_cuda': False
        }

        self.change_params(params_dict)

    def set(self, group, val_dict, set_if_not_exists=False, verbose=False):
        """ Add key-value pairs to a group. Existing key-value pairs will be overwritten
            if specified in val_dict, but not deleted.

        Args:
            group: The name of the group.
            val_dict: A dictionary with key-value pairs to be set for the group.
            set_if_not_exists: Whether to set a key-value pair in a group if the key does not currently exist in the group.
        """
        if not hasattr(self, group):
            raise KeyError('No group in VolParams named {0}'.format(group))

        d = getattr(self, group)
        for k, v in val_dict.items():
            if k not in d and not set_if_not_exists:
                if verbose:
                    logging.warning(
                        "NOT setting value of key {0} in group {1}, because no prior key existed...".format(k, group))
            else:
                if np.any(d[k] != v):
                    logging.warning(
                        "Changing key {0} in group {1} from {2} to {3}".format(k, group, d[k], v))
                setattr(d, k, v)

    def get(self, group, key):
        """ Get a value for a given group and key. Raises an exception if no such group/key combination exists.

        Args:
            group: The name of the group.
            key: The key for the property in the group of interest.

        Returns: The value for the group/key combination.
        """
        if not hasattr(self, group):
            raise KeyError('No group in VolParams named {0}'.format(group))

        d = getattr(self, group)
        if key not in d:
            raise KeyError('No key {0} in group {1}'.format(key, group))

        return d[key]

    def get_group(self, group):
        """ Get the dictionary of key-value pairs for a group.

        Args:
            group: The name of the group.
        """
        if not hasattr(self, group):
            raise KeyError('No group in VolParams named {0}'.format(group))

        return getattr(self, group)

    def change_params(self, params_dict, verbose=False):
        for gr in self.__dict__:
            self.set(gr, params_dict, verbose=verbose)
        for k, v in params_dict.items():
            flag = True
            for gr in self.__dict__:
                d = getattr(self, gr)
                if k in d:
                    flag = False
            if flag:
                logging.warning('No parameter {0} found!'.format(k))
        return self