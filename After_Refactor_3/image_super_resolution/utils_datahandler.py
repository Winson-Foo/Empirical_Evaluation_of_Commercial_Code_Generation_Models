import os
import imageio
import numpy as np
from CONSTANT.logger import get_logger
from typing import List, Dict, Tuple


class DataHandler:
    LR_DIR: str = 'lr'
    HR_DIR: str = 'hr'
    EXTENSIONS: Tuple[str, str, str] = ('.png', '.jpeg', '.jpg')
    AUGMENT_TRANSFORMS: int = 3

    def __init__(self, lr_dir: str, hr_dir: str, patch_size: int, scale: int, n_validation_samples: int = None):
        self.folders: Dict[str, str] = {self.HR_DIR: hr_dir, self.LR_DIR: lr_dir}
        self.img_list: Dict[str, np.ndarray] = {}
        self.n_validation_samples: int = n_validation_samples
        self.patch_size: Dict[str, int] = {self.LR_DIR: patch_size, self.HR_DIR: patch_size * scale}
        self.logger = get_logger(__name__)
        self._make_img_list()
        self._check_dataset()

    def _make_img_list(self) -> None:
        for res in [self.HR_DIR, self.LR_DIR]:
            file_names: List[str] = os.listdir(self.folders[res])
            file_names: List[str] = [file for file in file_names if file.endswith(self.EXTENSIONS)]
            self.img_list[res] = np.sort(file_names)
        if self.n_validation_samples:
            samples = np.random.choice(range(len(self.img_list[self.HR_DIR])), self.n_validation_samples, replace=False)
            for res in [self.HR_DIR, self.LR_DIR]:
                self.img_list[res] = self.img_list[res][samples]

    def _check_dataset(self) -> None:
        assert len(self.img_list[self.HR_DIR]) == self.img_list[self.HR_DIR].shape[0], 'UnevenDatasets'
        assert self._matching_datasets(), 'Input/LabelsMismatch'

    def _matching_datasets(self) -> bool:
        LR_name_root = [x.split('.')[0].rsplit('x', 1)[0] for x in self.img_list[self.LR_DIR]]
        HR_name_root = [x.split('.')[0] for x in self.img_list[self.HR_DIR]]
        return np.all(HR_name_root == LR_name_root)

    def _not_flat(self, patch: np.ndarray, flatness: float) -> bool:
        if max(np.std(patch, axis=0).mean(), np.std(patch, axis=1).mean()) < flatness:
            return False
        else:
            return True

    def _crop_imgs(self, imgs: Dict[str, np.ndarray], batch_size: int, flatness: float) -> Dict[str, np.ndarray]:
        slices: Dict[str, np.ndarray] = {}
        crops: Dict[str, List] = {self.LR_DIR: [], self.HR_DIR: []}
        accepted_slices: Dict[str, List[int]] = {self.LR_DIR: [], self.HR_DIR: []}
        top_left: Dict[str, Dict[str, np.ndarray]] = {'x': {}, 'y': {}}
        n: int = 50 * batch_size
        for i, axis in enumerate(['x', 'y']):
            top_left[axis][self.LR_DIR] = np.random.randint(0, imgs[self.LR_DIR].shape[i] - self.patch_size[self.LR_DIR] + 1, batch_size + n)
            top_left[axis][self.HR_DIR] = top_left[axis][self.LR_DIR] * self.patch_size[self.HR_DIR]
        for res in [self.LR_DIR, self.HR_DIR]:
            slices[res] = np.array([{'x': (x, x + self.patch_size[res]), 'y': (y, y + self.patch_size[res])} for x, y in zip(top_left['x'][res], top_left['y'][res])])

        for slice_index, s in enumerate(slices[self.LR_DIR]):
            candidate_crop = imgs[self.LR_DIR][s['x'][0]: s['x'][1], s['y'][0]: s['y'][1], slice(None)]
            if self._not_flat(candidate_crop, flatness) or n == 0:
                crops[self.LR_DIR].append(candidate_crop)
                accepted_slices[self.LR_DIR].append(slice_index)
            else:
                n -= 1
            if len(crops[self.LR_DIR]) == batch_size:
                break

        accepted_slices[self.HR_DIR] = slices[self.HR_DIR][accepted_slices[self.LR_DIR]]
        for s in accepted_slices[self.HR_DIR]:
            candidate_crop = imgs[self.HR_DIR][s['x'][0]: s['x'][1], s['y'][0]: s['y'][1], slice(None)]
            crops[self.HR_DIR].append(candidate_crop)

        crops[self.LR_DIR] = np.array(crops[self.LR_DIR])
        crops[self.HR_DIR] = np.array(crops[self.HR_DIR])
        return crops

    def _apply_transform(self, img: np.ndarray, transform_selection: Tuple[int, int]) -> np.ndarray:
        rotate = {
            0: lambda x: x,
            1: lambda x: np.rot90(x, k=1, axes=(1, 0)), # rotate right
            2: lambda x: np.rot90(x, k=1, axes=(0, 1)) # rotate left
        }

        flip = {
            0: lambda x: x,
            1: lambda x: np.flip(x, 0), # flip along horizontal axis
            2: lambda x: np.flip(x, 1) # flip along vertical axis
        }

        rot_direction, flip_axis = transform_selection
        img = rotate[rot_direction](img)
        img = flip[flip_axis](img)
        return img

    def _transform_batch(self, batch: np.ndarray, transforms: np.ndarray) -> np.ndarray:
        t_batch = np.array([self._apply_transform(img, transforms[i]) for i, img in enumerate(batch)])
        return t_batch

    def get_batch(self, batch_size: int, idx: int = None, flatness: float = 0.0) -> Dict[str, np.ndarray]:
        if not idx:
            idx = np.random.choice(range(len(self.img_list[self.HR_DIR])))
        img = {}
        for res in [self.LR_DIR, self.HR_DIR]:
            img_path = os.path.join(self.folders[res], self.img_list[res][idx])
            img[res] = imageio.imread(img_path) / 255.0
        batch = self._crop_imgs(img, batch_size, flatness)
        transforms = np.random.randint(0, self.AUGMENT_TRANSFORMS, (batch_size, 2))
        batch[self.LR_DIR] = self._transform_batch(batch[self.LR_DIR], transforms)
        batch[self.HR_DIR] = self._transform_batch(batch[self.HR_DIR], transforms)
        return batch

    def get_validation_batches(self, batch_size: int) -> List[Dict[str, np.ndarray]]:
        if self.n_validation_samples:
            batches = []
            for idx in range(self.n_validation_samples):
                batches.append(self.get_batch(batch_size, idx, flatness=0.0))
            return batches
        else:
            self.logger.error('No validation set size specified. (not operating in a validation set?)')
            raise ValueError('No validation set size specified. (not operating in a validation set?)')

    def get_validation_set(self, batch_size: int) -> Dict[str, np.ndarray]:
        if self.n_validation_samples:
            batches = self.get_validation_batches(batch_size)
            valid_set: Dict[str, List[np.ndarray]] = {self.LR_DIR: [], self.HR_DIR: []}
            for batch in batches:
                for res in (self.LR_DIR, self.HR_DIR):
                    valid_set[res].extend(batch[res])
            for res in (self.LR_DIR, self.HR_DIR):
                valid_set[res] = np.array(valid_set[res])
            return valid_set
        else:
            self.logger.error('No validation set size specified. (not operating in a validation set?)')
            raise ValueError('No validation set size specified. (not operating in a validation set?)')