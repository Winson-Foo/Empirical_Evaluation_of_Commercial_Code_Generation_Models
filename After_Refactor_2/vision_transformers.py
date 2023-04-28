from typing import Optional, List
import numpy as np
import cv2

from ...core.node import ProcessorNode
from ...utils.transforms import resize_add_padding

class CropImageTransformer(ProcessorNode):
    def __init__(self, crop_dimensions: Optional[np.array] = None):
        self.crop_dimensions = None
        self.set_crop_dimensions(crop_dimensions)
        super(CropImageTransformer, self).__init__()

    def set_crop_dimensions(self, crop_dimensions: Optional[np.array]):
        if crop_dimensions is not None:
            self._check_crop_dimensions(crop_dimensions)
        self.crop_dimensions = crop_dimensions

    def _check_crop_dimensions(self, crop_dimensions: np.array):
        if (crop_dimensions < 0).any():
            raise ValueError('One of the crop values is less than 0')
        if ((crop_dimensions[:, 0] > crop_dimensions[:, 2]).any()
            or (crop_dimensions[:, 1] > crop_dimensions[:, 3]).any()):
            raise ValueError('ymin > ymax or xmin > xmax')

    def _crop(self, im: np.array, crop_dimensions: Optional[np.array] = None) -> List[np.array]:
        if crop_dimensions is None:
            if self.crop_dimensions is None:
                raise RuntimeError("Crop dimensions were not specified")
            crop_dimensions = self.crop_dimensions
        self._check_crop_dimensions(crop_dimensions)
        if ((crop_dimensions[:, 0] > im.shape[0]).any()
            or (crop_dimensions[:, 2] > im.shape[1]).any()):
            raise ValueError('One of the crop indexes is out of bounds')
        result = []
        for crop_dimensions_x in crop_dimensions:
            ymin, ymax = int(crop_dimensions_x[0]), int(crop_dimensions_x[2])
            xmin, xmax = int(crop_dimensions_x[1]), int(crop_dimensions_x[3])
            im_cropped = im[ymin:ymax, xmin:xmax, :]
            result.append(im_cropped)
        return result
    
    def process(self, im: np.array, crop_dimensions: Optional[np.array]) -> List[np.array]:
        to_transform = np.array(im)
        if crop_dimensions is not None:
            self.set_crop_dimensions(crop_dimensions)
        return self._crop(to_transform, self.crop_dimensions)


class MaskImageTransformer(ProcessorNode):
    def __init__(self):
        super(MaskImageTransformer, self).__init__()
    
    def process(self, im : np.array, mask : np.array) -> np.array:
        if mask.shape[:2] != im.shape[:2]:
            raise ValueError("`mask` does not have same dimensions as `im`")
        to_transform = np.array(im)
        im = im.astype(float)
        alpha = cv2.merge((mask, mask, mask))
        masked = cv2.multiply(im, alpha)
        return masked.astype(np.uint8)


class ResizeImageTransformer(ProcessorNode):
    def __init__(self, maintain_ratio = False):
        self._maintain_ratio = maintain_ratio
        super(ResizeImageTransformer, self).__init__()

    def _resize(self, im : np.array, new_size) -> np.array:
        height, width = new_size
        if height < 0 or width < 0:
            raise ValueError("One of `width` or `height` is a negative value")
        if self._maintain_ratio:
            im = resize_add_padding(im, height, width)
        else:
            im = cv2.resize(im, (width, height))
        return im
    
    def process(self, im : np.array, new_size) -> np.array:
        to_transform = np.array(im)
        if not isinstance(new_size, tuple) or len(new_size) != 2:
            raise ValueError("new_size should be a tuple of (height, width)")
        return self._resize(to_transform, new_size)