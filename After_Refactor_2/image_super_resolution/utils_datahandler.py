import os
import imageio
import numpy as np
from CONSTANT.logger import get_logger

class ImageDataHandler:
    """
    Generates augmented batches used for training or validation.
    
    Args:
        lr_dir: directory containing the Low Res images.
        hr_dir: directory containing the High Res images.
        patch_size: integer, size of the patches extracted from LR images.
        scale: integer, upscaling factor.
        n_validation_samples: integer, size of the validation set. Only provided if the
            DataHandler is used to generate validation sets.
    """
    def __init__(self, lr_dir, hr_dir, patch_size, scale, n_validation_samples=None):
        self.folders = {'hr': hr_dir, 'lr': lr_dir}
        self.img_list = {} # list of file names
        self.n_validation_samples = n_validation_samples
        self.patch_size = patch_size
        self.scale = scale * patch_size
        self.logger = get_logger(__name__)
        self._create_image_list()
        self._verify_dataset()

    def _create_image_list(self):
        """
        Creates a dictionary of lists of the acceptable images contained in lr_dir and hr_dir.
        """
        for res in ['hr', 'lr']:
            self.img_list[res] = sorted([file for file in os.listdir(self.folders[res]) if file.endswith('.png') or file.endswith('.jpeg') or file.endswith('.jpg')])

        if self.n_validation_samples:
            samples = np.random.choice(len(self.img_list['hr']), self.n_validation_samples, replace=False)
            for res in ['hr', 'lr']:
                self.img_list[res] = self.img_list[res][samples]

    def _verify_dataset(self):
        """
        Sanity check for dataset.
        """
        assert len(self.img_list['hr']) == len(self.img_list['lr']), 'Input/LabelsMismatch'
        
    def _match_image_sets(self):
        """
        Returns file names matching between lr and hr directories.
        """
        # LR_name.png = HR_name+x+scale.png
        LR_name_root = [x.split('.')[0] for x in self.img_list['lr']]
        HR_name_root = [x.split('.')[0].strip(f"+{self.scale}") for x in self.img_list['hr']]
        return np.all(HR_name_root == LR_name_root)

    def _is_complex_patch(self, patch, flatness):
        """
        Determines whether the patch is complex, or not-flat enough.
        Threshold set by flatness.
        """
        if max(np.std(patch, axis=0).mean(), np.std(patch, axis=1).mean()) < flatness:
            return False
        return True
    
    def _crop_imgs(self, imgs, batch_size, flatness):
        """
        Get random top left corners coordinates in LR space, multiply by scale to
        get HR coordinates.
        Gets batch_size + n possible coordinates.
        Accepts the batch only if the standard deviation of pixel intensities is above a given threshold, OR
        no patches can be further discarded (n have been discarded already).
        Square crops of size patch_size are taken from the selected
        top left corners.
        """
        n = 0
        crops = {'lr':[], 'hr':[]}
        while len(crops['lr']) < batch_size and n <= 50 * batch_size :
            top_left = {'x': {}, 'y': {}}
            slices = {}
            accepted_slices = {}
            accepted_slices['lr'] = []
            for i, axis in enumerate(['x', 'y']):
                top_left[axis]['lr'] = np.random.randint(0, imgs['lr'].shape[i] - self.patch_size + 1, batch_size + n)
                top_left[axis]['hr'] = top_left[axis]['lr'] * self.scale
            for res in ['lr', 'hr']:
                slices[res] = np.array(
                    [
                        {'x': (x, x + self.patch_size), 'y': (y, y + self.patch_size)}
                        for x, y in zip(top_left['x'][res], top_left['y'][res])
                    ]
                )

            for slice_index, s in enumerate(slices['lr']):
                candidate_crop = imgs['lr'][s['x'][0]: s['x'][1], s['y'][0]: s['y'][1], slice(None)]
                if self._is_complex_patch(candidate_crop, flatness) or n == 50 * batch_size:
                    crops['lr'].append(candidate_crop)
                    accepted_slices['lr'].append(slice_index)
                else:
                    n += 1
                    
            accepted_slices['hr'] = slices['hr'][accepted_slices['lr']]
            for s in accepted_slices['hr']:
                candidate_crop = imgs['hr'][s['x'][0]: s['x'][1], s['y'][0]: s['y'][1], slice(None)]
                crops['hr'].append(candidate_crop)
        return  {'lr': np.array(crops['lr'][:batch_size]), 'hr': np.array(crops['hr'][:batch_size])}
    
    @staticmethod
    def _apply_transform(img, transform_selection):
        """ Rotates and flips input image according to transform_selection. """
        
        rotate = {0: lambda x: x, 1: lambda x: np.rot90(x, k=1, axes=(1, 0)), 2: lambda x: np.rot90(x, k=1, axes=(0, 1))}
        flip = {0: lambda x: x, 1: lambda x: np.flip(x, 0), 2: lambda x: np.flip(x, 1)}
        
        img = rotate[transform_selection[0]](img)
        img = flip[transform_selection[1]](img)
        return img
    
    def _transform_batch(self, batch, transforms):
        """ Transforms each individual image of the batch independently. """
        return np.array([self._apply_transform(img, transforms[i]) for i, img in enumerate(batch)])
  
    def get_batch(self, batch_size, idx=None, flatness=0.0):
        """
        Returns a dictionary with keys ('lr', 'hr') containing training batches
        of Low Res and High Res image patches.

        Args:
            batch_size: integer.
            flatness: float in [0,1], is the patch "flatness" threshold.
                Determines what level of detail the patches need to meet. 0 means any patch is accepted.
        """
        if not idx:
            idx = np.random.choice(len(self.img_list['hr']))
        img = {res: imageio.imread(os.path.join(self.folders[res], self.img_list[res][idx])) / 255.0 for res in ['lr', 'hr']}
        crops = self._crop_imgs(img, batch_size, flatness)
        transforms = np.random.randint(0, 3, (batch_size, 2))
        return {'lr': self._transform_batch(crops['lr'], transforms), 
                'hr': self._transform_batch(crops['hr'], transforms)}


    def get_validation_batches(self, batch_size, flatness=0.0):
        """
        Returns a batch for each image in the validation set.
        """
        if self.n_validation_samples:
            return [self.get_batch(batch_size, idx=i, flatness=flatness) for i in range(self.n_validation_samples)]
        self.logger.error('No validation set size specified. (not operating in a validation set?)')
        raise ValueError('No validation set size specified. (not operating in a validation set?)')


    def get_validation_set(self, batch_size, flatness=0.0):
        """
        Returns a batch for each image in the validation set.
        Flattens and splits them to feed it to Keras's model.evaluate.
        """
        if self.n_validation_samples:
            batches = self.get_validation_batches(batch_size, flatness=flatness)
            valid_set = {'lr': [], 'hr': []}
            for batch in batches:
                for res in ('lr', 'hr'):
                    valid_set[res].extend(batch[res])
            return {'lr': np.array(valid_set['lr']), 'hr': np.array(valid_set['hr'])}
        self.logger.error('No validation set size specified. (not operating in a validation set?)')
        raise ValueError('No validation set size specified. (not operating in a validation set?)')