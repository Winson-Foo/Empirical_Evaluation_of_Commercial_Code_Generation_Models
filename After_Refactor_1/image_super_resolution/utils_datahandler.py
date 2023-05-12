import os
import random 

import imageio
import numpy as np

from ISR.utils.logger import get_logger


class DataHandler:
    """
    DataHandler generate augmented batches used for training or validation.

    Args:
        lr_dir: directory containing the Low Res images.
        hr_dir: directory containing the High Res images.
        patch_size: integer, size of the patches extracted from LR images.
        scale: integer, upscaling factor.
        n_validation_samples: integer, size of the validation set. Only provided if the
            DataHandler is used to generate validation sets.
    """
    
    def __init__(self, lr_dir, hr_dir, patch_size, scale, n_validation_samples=None):
        self.image_folders = {'hr': hr_dir, 'lr': lr_dir}  
        self.admissible_extensions = ('.png', '.jpeg', '.jpg')  
        self.image_list = {}  
        self.n_validation_samples = n_validation_samples
        self.patch_size = patch_size
        self.scale = scale
        self.patch_size = {'lr': patch_size, 'hr': patch_size * self.scale}
        self.logger = get_logger(__name__)
        self._create_image_list()
        self.sanity_check_data()
    
    def _create_image_list(self):
        """
        Creates a dictionary of lists of the acceptable images contained in lr_dir and hr_dir. 
        Returns a dictionary of images. 
        """
        for res in ['hr', 'lr']:
            file_names = os.listdir(self.image_folders[res])
            file_names = [file for file in file_names if file.endswith(self.admissible_extensions)]
            self.image_list[res] = np.sort(file_names)
        
        if self.n_validation_samples:
            validation_samples = np.random.choice(
                range(len(self.image_list['hr'])), self.n_validation_samples, replace=False
            )
            for res in ['hr', 'lr']:
                self.image_list[res] = self.image_list[res][validation_samples]
    
    def sanity_check_data(self):
        """ 
        Asserts that the datasets are even and input and labels have a matching file name. 
        """
        
        # the order of these asserts is important for testing
        assert len(self.image_list['hr']) == self.image_list['hr'].shape[0], 'UnevenDatasets'
        assert self.matching_datasets(), 'Input/LabelsMismatch'
    
    def matching_datasets(self):
        """ 
        Rough file name matching between lr and hr directories.
        LR_name.png = HR_name+x+scale.png
        OR
        LR_name.png = HR_name.png
        LR_name_root contains the root name and HR_name_root contains root name and scale of filename for each directory.
        Returns True or False. 
        """
        LR_name_root = [x.split('.')[0].rsplit('x', 1)[0] for x in self.image_list['lr']]
        HR_name_root = [x.split('.')[0] for x in self.image_list['hr']]
        return np.all(HR_name_root == LR_name_root)
    
    def not_flat(self, patch, flatness):
        """
        Determines whether the patch is complex, or not-flat enough.
        Threshold set by flatness.
        Returns boolean. 
        """
        
        if max(np.std(patch, axis=0).mean(), np.std(patch, axis=1).mean()) < flatness:
            return False
        else:
            return True
    
    def create_random_top_left_coordinates(self, batch_size, image_dimensions):
        """
        Get random top left corners coordinates in LR space, multiply by scale to get HR coordinates.
        Returns dictionary of top_left coordinates for each resolution. 
        """
        top_left = {'x': {}, 'y': {}}
        for i, axis in enumerate(['x', 'y']):
            top_left[axis]['lr'] = np.random.randint(0, image_dimensions['lr'][i] - self.patch_size['lr'] + 1, batch_size)
            top_left[axis]['hr'] = top_left[axis]['lr'] * self.scale
        return top_left 
    
    def create_slices(self, top_left):
        """
        Square crops of size patch_size are taken from the selected top left corners.
        Returns a dictionary of coordinate slices for each resolution. 
        """
        slices = {'lr': [], 'hr': []}
        for res in ['lr', 'hr']:
            slices[res] = np.array(
                [
                    {'x': (x, x + self.patch_size[res]), 'y': (y, y + self.patch_size[res])}
                    for x, y in zip(top_left['x'][res], top_left['y'][res])
                ]
            )
        return slices
    
    def get_image_crops(self, image, slices, batch_size, flatness):
        """
        Accepts the batch only if the standard deviation of pixel intensities is above a given threshold, OR
        no patches can be further discarded (n have been discarded already).
        Returns dictionary of accepted images for each resolution. 
        """
        crops = {'lr': [] , 'hr': []}
        accepted_slices = {'lr': []}
        n = 50 * batch_size
        for slice_index, s in enumerate(slices['lr']):
            candidate_crop = image['lr'][s['x'][0]: s['x'][1], s['y'][0]: s['y'][1], slice(None)]
            if self.not_flat(candidate_crop, flatness) or n == 0:
                crops['lr'].append(candidate_crop)
                accepted_slices['lr'].append(slice_index)
            else:
                n -= 1
            if len(crops['lr']) == batch_size:
                break
        
        accepted_slices['hr'] = slices['hr'][accepted_slices['lr']]
        
        for s in accepted_slices['hr']:
            candidate_crop = image['hr'][s['x'][0]: s['x'][1], s['y'][0]: s['y'][1], slice(None)]
            crops['hr'].append(candidate_crop)
        
        crops['lr'] = np.array(crops['lr'])
        crops['hr'] = np.array(crops['hr'])
        return crops
    
    def apply_transform(self, image, transform_selection):
        """ Rotates and flips input image according to transform_selection. """
        
        rotate = {
            0: lambda x: x,
            1: lambda x: np.rot90(x, k=1, axes=(1, 0)),  # rotate right
            2: lambda x: np.rot90(x, k=1, axes=(0, 1)),  # rotate left
        }
        
        flip = {
            0: lambda x: x,
            1: lambda x: np.flip(x, 0),  # flip along horizontal axis
            2: lambda x: np.flip(x, 1),  # flip along vertical axis
        }
        
        rot_direction = transform_selection[0]
        flip_axis = transform_selection[1]
        
        image = rotate[rot_direction](image)
        image = flip[flip_axis](image)
        
        return image
    
    def transform_batch(self, batch, transforms):
        """ 
        Transforms each individual image of the batch independently and returns the transformed batch. 
        """
        t_batch = np.array(
            [self.apply_transform(img, transforms[i]) for i, img in enumerate(batch)]
        )
        return t_batch
    
    def generate_training_batches(self, batch_size, idx=None, flatness=0.0):
        """
        Returns a dictionary with keys ('lr', 'hr') containing training batches
        of Low Res and High Res image patches.

        Args:
            batch_size: integer.
            flatness: float in [0,1], is the patch "flatness" threshold.
                Determines what level of detail the patches need to meet. 0 means any patch is accepted.
        """
        if not idx:
            # randomly select one image. idx is passed at validation time.
            idx = random.randint(0, len(self.image_list['hr']) - 1) 
        image = {}
        for res in ['lr', 'hr']:
            img_path = os.path.join(self.image_folders[res], self.image_list[res][idx])
            image[res] = imageio.imread(img_path) / 255.0
        top_left_coordinates = self.create_random_top_left_coordinates(batch_size, {'lr': image['lr'].shape[:-1],'hr': image['hr'].shape[:-1]})
        slices = self.create_slices(top_left_coordinates)
        batch = self.get_image_crops(image, slices, batch_size, flatness)
        transforms = np.random.randint(0, 3, (batch_size, 2))
        batch['lr'] = self.transform_batch(batch['lr'], transforms)
        batch['hr'] = self.transform_batch(batch['hr'], transforms)
        return batch
    
    def generate_validation_batches(self, batch_size):
        """ 
        Returns a batch for each image in the validation set. 
        """
        if self.n_validation_samples:
            batches = []
            for idx in range(self.n_validation_samples):
                batches.append(self.generate_training_batches(batch_size, idx, flatness=0.0))
            return batches
        else:
            self.logger.error(
                'No validation set size specified. (not operating in a validation set?)'
            )
            raise ValueError(
                'No validation set size specified. (not operating in a validation set?)'
            )