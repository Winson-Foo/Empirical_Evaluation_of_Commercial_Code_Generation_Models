from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ...core.node import ProcessorNode
from ...utils.transforms import resize_add_padding


class CropImageTransformer(ProcessorNode):
    """
    This class crops the image according to the coordinates in crop_dimensions.

    Raises:
        - ValueError:
            - If any of crop_dimensions less than 0
            - If ymin > ymax or xmin > xmax
    """
    def __init__(self, crop_dimensions: Optional[np.ndarray] = None):
        """
        Args:
            crop_dimensions: np.array of shape (nb_boxes, 4),
                second dimension entries are [ymin, xmin, ymax, xmax], or None
        """
        super().__init__()
        if crop_dimensions is not None:
            self._check_crop_dimensions(crop_dimensions)
            self.crop_dimensions = crop_dimensions

    @staticmethod
    def _check_crop_dimensions(crop_dimensions: np.ndarray) -> None:
        """
        Checks if crop dimensions are valid.

        Raises:
            - ValueError:
                - If any of crop_dimensions less than 0
                - If ymin > ymax or xmin > xmax
        """
        if (crop_dimensions < 0).any():
            raise ValueError("One of the crop values is less than 0")
        if ((crop_dimensions[:, 0] > crop_dimensions[:, 2]).any()
                or (crop_dimensions[:, 1] > crop_dimensions[:, 3]).any()):
            raise ValueError("ymin > ymax or xmin > xmax")

    def _crop(self, image: np.ndarray, crop_dimensions: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Crops image according to the coordinates in crop_dimensions.
        If those coordinates are out of bounds, it will raise errors.

        Args:
            im: Image to be cropped with the shape of (height, width, channels)
            crop_dimensions: np.array of shape (nb_boxes, 4),
                second dimension entries are [ymin, xmin, ymax, xmax],
                or None
        Raises:
            - ValueError:
                - If any of crop_dimensions less than 0
                - If any of crop_dimensions out of bounds
                - If ymin > ymax or xmin > xmax
        Returns:
            - List of np.arrays: Returns a list of cropped images of the same size as crop_dimensions
        """
        if crop_dimensions is None:
            if self.crop_dimensions is None:
                raise RuntimeError("Crop dimensions were not specified")
            crop_dimensions = self.crop_dimensions
        self._check_crop_dimensions(crop_dimensions)

        if ((crop_dimensions[:, 0] > image.shape[0]).any()
                or (crop_dimensions[:, 2] > image.shape[1]).any()):
            raise ValueError("One of the crop indexes is out of bounds")
      
        crops = [image[int(ymin):int(ymax), int(xmin):int(xmax), :]
                 for ymin, xmin, ymax, xmax in crop_dimensions]

        return crops

    def process(self, image: np.ndarray, crop_dimensions: Optional[np.ndarray]) -> List[np.array]:
        """
        Crops an image according to the coordinates in crop_dimensions.
        If those coordinates are out of bounds, it will raise errors.

        Args:
            image: Image to be cropped with the shape of (height, width, channels)
            crop_dimensions: np.array of shape (nb_boxes, 4),
                second dimension entries are [ymin, xmin, ymax, xmax],
                or None
        Raises:
            - ValueError:
                - If any of crop_dimensions less than 0
                - If any of crop_dimensions out of bounds
                - If ymin > ymax or xmin > xmax
        Returns:
            - List of np.arrays: Returns a list of cropped images of the same size as crop_dimensions
        """
        return self._crop(np.array(image), crop_dimensions)


class MaskImageTransformer(ProcessorNode):
    """
    Masks an image according to the given mask.

    Raises:
        - ValueError:
            - If ``mask`` does not have same height and width as ``image``
    """
    def __init__(self):
        super().__init__()

    def _mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Masks an image according to given masks.

        Args:
            image: Image to be cropped with the shape of (height, width, channels)
            mask: (height, width) of type np.float32, with values between zero and one
        Raises:
            ValueError: If ``mask`` does not have same height and width as ``image``
        """
        if mask.shape[:2] != image.shape[:2]:
            raise ValueError("`mask` does not have same dimensions as `im`")
        image = image.astype(float)
        alpha = cv2.merge((mask, mask, mask))

        masked = cv2.multiply(image, alpha)
        return masked.astype(np.uint8)

    def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Masks an image according to given masks.

        Args:
            image: Image to be cropped with the shape of (height, width, channels)
            mask: (height, width) of type np.float32, with values between zero and one
        Raises:
            ValueError: If ``mask`` does not have same height and width as ``image``
        """
        return self._mask(np.array(image), np.array(mask))


class ResizeImageTransformer(ProcessorNode):
    """
    Resizes an image according to the given size.

    Raises:
        - ValueError:
            - If ``new_height`` or ``new_width`` are negative
    """
    def __init__(self, maintain_ratio: bool = False):
        """
        Args:
            maintain_ratio: if True, it maintains the aspect ratio of the image while resizing.
        """
        super().__init__()
        self._maintain_ratio = maintain_ratio

    def _resize(self, image: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
        """
        Resizes image according to coordinates in new_size.

        Args:
            image: Image to be resized with the shape of (height, width, channels)
            new_size: A tuple of (new_height, new_width)
        Raises:
            ValueError: If ``new_height`` or ``new_width`` are negative.
        """
        new_height, new_width = new_size
        if new_height < 0 or new_width < 0:
            raise ValueError("One of `width` or `height` is a negative value")

        if self._maintain_ratio:
            image = resize_add_padding(image, new_height, new_width)
        else:
            image = cv2.resize(image, (new_width, new_height))

        return image

    def process(self, image: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
        """
        Resizes image according to coordinates in new_size.

        Args:
            image: Image to be resized with the shape of (height, width, channels)
            new_size: A tuple of (new_height, new_width)
        Raises:
            ValueError: If ``new_height`` or ``new_width`` are negative.
        """
        return self._resize(np.array(image), new_size)
