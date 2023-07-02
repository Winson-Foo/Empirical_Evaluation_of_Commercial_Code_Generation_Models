import logging
import numpy as np

class VolParams(object):
    """Class for setting parameters for voltage imaging."""

    def __init__(self, data_params=None, volspike_params=None, motion_params=None, params_dict={}):
        """
        Initialize the VolParams object with the specified parameter groups.
        
        Args:
            data_params: A dictionary of parameters for data processing.
            volspike_params: A dictionary of parameters for volspike processing.
            motion_params: A dictionary of parameters for motion correction.
            params_dict: A dictionary containing all the parameter groups.
        """
        self.data = VolParamsData(data_params)
        self.volspike = VolParamsVolSpike(volspike_params)
        self.motion = VolParamsMotion(motion_params)
        self.change_params(params_dict)

    def set(self, group, val_dict, set_if_not_exists=False, verbose=False):
        """
        Add key-value pairs to a specified group.

        Args:
            group: The name of the group.
            val_dict: A dictionary with key-value pairs to be set for the group.
            set_if_not_exists: Whether to set a key-value pair in a group if the key does not currently exist.
            verbose: Whether to print warnings for changes to existing key-value pairs.
        """
        if not hasattr(self, group):
            raise KeyError('No group in VolParams named {0}'.format(group))

        d = getattr(self, group)
        for k, v in val_dict.items():
            if k not in d and not set_if_not_exists:
                if verbose:
                    logging.warning(
                        "NOT setting value of key {0} in group {1}, because no prior key existed...".format(k, group)
                    )
            else:
                if np.any(d[k] != v):
                    logging.warning(
                        "Changing key {0} in group {1} from {2} to {3}".format(k, group, d[k], v)
                    )
                d[k] = v

    def get(self, group, key):
        """
        Get a value for a given group and key.

        Args:
            group: The name of the group.
            key: The key for the property in the group of interest.

        Returns:
            The value for the group/key combination.
        """
        if not hasattr(self, group):
            raise KeyError('No group in VolParams named {0}'.format(group))

        d = getattr(self, group)
        if key not in d:
            raise KeyError('No key {0} in group {1}'.format(key, group))

        return d[key]

    def get_group(self, group):
        """
        Get the dictionary of key-value pairs for a group.

        Args:
            group: The name of the group.
        """
        if not hasattr(self, group):
            raise KeyError('No group in VolParams named {0}'.format(group))

        return getattr(self, group)

    def change_params(self, params_dict, verbose=False):
        """
        Change parameter values based on the specified dictionary.

        Args:
            params_dict: A dictionary containing the new parameter values.
            verbose: Whether to print warnings for parameters not found.
        """
        for gr in list(self.__dict__.keys()):
            self.set(gr, params_dict, verbose=verbose)

        for k, v in params_dict.items():
            flag = True
            for gr in list(self.__dict__.keys()):
                d = getattr(self, gr)
                if k in d:
                    flag = False
            if flag:
                logging.warning('No parameter {0} found!'.format(k))

        return self

class VolParamsData(object):
    """Class for setting parameters related to data processing."""

    def __init__(self, fnames=None, fr=None, index=None, ROIs=None, weights=None):
        self.fnames = fnames
        self.fr = fr
        self.index = index
        self.ROIs = ROIs
        self.weights = weights

class VolParamsVolSpike(object):
    """Class for setting parameters related to volspike processing."""

    def __init__(self, template_size=0.02, context_size=35, censor_size=12, visualize_ROI=False, flip_signal=True,
                 hp_freq_pb=1/3, nPC_bg=8, ridge_bg=0.01, hp_freq=1, clip=100, threshold_method='adaptive_threshold',
                 min_spikes=10, pnorm=0.5, threshold=3, sigmas=np.array([1, 1.5, 2]), n_iter=2, weight_update='ridge',
                 do_plot=False, do_cross_val=False, sub_freq=20, method='spikepursuit', superfactor=10):
        self.template_size = template_size
        self.context_size = context_size
        self.censor_size = censor_size
        self.visualize_ROI = visualize_ROI
        self.flip_signal = flip_signal
        self.hp_freq_pb = hp_freq_pb
        self.nPC_bg = nPC_bg
        self.ridge_bg = ridge_bg
        self.hp_freq = hp_freq
        self.clip = clip
        self.threshold_method = threshold_method
        self.min_spikes = min_spikes
        self.pnorm = pnorm
        self.threshold = threshold
        self.sigmas = sigmas
        self.n_iter = n_iter
        self.weight_update = weight_update
        self.do_plot = do_plot
        self.do_cross_val = do_cross_val
        self.sub_freq = sub_freq
        self.method = method
        self.superfactor = superfactor

class VolParamsMotion(object):
    """Class for setting parameters related to motion correction."""

    def __init__(self, border_nan='copy', gSig_filt=None, max_deviation_rigid=3, max_shifts=(6, 6),
                 min_mov=None, niter_rig=1, nonneg_movie=True, num_frames_split=80, num_splits_to_process_els=None,
                 num_splits_to_process_rig=None, overlaps=(32, 32), pw_rigid=False, shifts_opencv=True,
                 splits_els=14, splits_rig=14, strides=(96, 96), upsample_factor_grid=4, use_cuda=False):
        self.border_nan = border_nan
        self.gSig_filt = gSig_filt
        self.max_deviation_rigid = max_deviation_rigid
        self.max_shifts = max_shifts
        self.min_mov = min_mov
        self.niter_rig = niter_rig
        self.nonneg_movie = nonneg_movie
        self.num_frames_split = num_frames_split
        self.num_splits_to_process_els = num_splits_to_process_els
        self.num_splits_to_process_rig = num_splits_to_process_rig
        self.overlaps = overlaps
        self.pw_rigid = pw_rigid
        self.shifts_opencv = shifts_opencv
        self.splits_els = splits_els
        self.splits_rig = splits_rig
        self.strides = strides
        self.upsample_factor_grid = upsample_factor_grid
        self.use_cuda = use_cuda