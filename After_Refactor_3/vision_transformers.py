from typing import Optional, List
import numpy as np
import cv2

from ...core.node import ProcessorNode
from ...utils.transforms import resize_add_padding


class ImageCropper(ProcessorNode):
    '''
    - Arguments:
        - crop_coords: np.array of shape (nb_boxes, 4) \
                second dimension entries are [ymin, xmin, ymax, xmax] \
                or None

    - Raises:
        - ValueError:
            - If any of crop_coords less than 0
            - If ymin > ymax or xmin > xmax
    '''

    def __init__(self, crop_coords: Optional[np.array] = None):
        if crop_coords:
            self._check_crop_coords(crop_coords)
            self.crop_coords = crop_coords
        super(ImageCropper, self).__init__()

    @staticmethod
    def _check_crop_coords(crop_coords: np.array):
        '''
        - Arguments:
            - crop_coords: np.array of shape (nb_boxes, 4) \
                    second dimension entries are [ymin, xmin, ymax, xmax]

        - Raises:
            - ValueError:
                - If any of crop_coords less than 0
                - If ymin > ymax or xmin > xmax
        '''
        if (crop_coords < 0).any():
            raise ValueError('One of the crop values is less than 0')
        if ((crop_coords[:, 0] > crop_coords[:, 2]).any()
                or (crop_coords[:, 1] > crop_coords[:, 3]).any()):
            raise ValueError('ymin > ymax or xmin > xmax')

    def _crop_image(self, image: np.array, crop_coords: Optional[np.array] = None) -> List[np.array]:
        '''
        - Arguments:
            - image (np.array): shape of (h, w, 3)
            - crop_coords: np.array of shape (nb_boxes, 4) \
                    second dimension entries are [ymin, xmin, ymax, xmax] \
                    or None

        - Raises:
            - ValueError:
                - If any of crop_coords less than 0
                - If any of crop_coords out of bounds
                - If ymin > ymax or xmin > xmax

        - Returns:
            - list of np.arrays: Returns a list of cropped images of the same size as crop_coords
        '''
        if crop_coords is None:
            if self.crop_coords is None:
                raise RuntimeError("Crop coords were not specified")
            crop_coords = self.crop_coords
        self._check_crop_coords(crop_coords)

        if ((crop_coords[:, 0] > image.shape[0]).any()
                or (crop_coords[:, 2] > image.shape[1]).any()):
            raise ValueError('One of the crop indexes is out of bounds')

        result = []
        for crop in crop_coords:
            ymin, ymax = int(crop[0]), int(crop[2])
            xmin, xmax = int(crop[1]), int(crop[3])
            cropped_image = image[ymin:ymax, xmin:xmax, :]
            result.append(cropped_image)
        return result
    
    def process(self, image: np.array, crop_coords: Optional[np.array]) -> List[np.array]:
        '''
        Crops image using the coordinates in crop_coords.
        If those coordinates are out of bounds, it will raise errors

        - Arguments:
            - image (np.array): shape of (h, w, 3)
            - crop_coords: np.array of shape (nb_boxes, 4) \
                    second dimension entries are [ymin, xmin, ymax, xmax] \
                    or None

        - Raises:
            - ValueError:
                - If any of crop_coords less than 0
                - If any of crop_coords out of bounds
                - If ymin > ymax or xmin > xmax

        - Returns:
            - list of np.arrays: Returns a list of cropped images of the same size as crop_coords
        '''
        to_transform = np.array(image)
        return self._crop_image(to_transform, crop_coords)


class ImageMasker(ProcessorNode):
    def __init__(self):
        super(ImageMasker, self).__init__()
    
    def _apply_mask(self, image: np.array, mask : np.array) -> np.array:
        if mask.shape[:2] != image.shape[:2]:
            raise ValueError("`mask` does not have same dimensions as `image`")
        image = image.astype(float)
        alpha = cv2.merge((mask, mask, mask))
        masked_image = cv2.multiply(image, alpha)
        return masked_image.astype(np.uint8)
    
    def process(self, image: np.array, mask : np.array) -> np.array:
        '''
        Masks an image according to the given mask

        - Arguments:
            - image (np.array): shape of (h, w, 3)
            - mask (np.array): (h, w) of type np.float32, with \
                values between zero and one
        
        - Raises:
            - ValueError:
                - If ``mask`` does not have the same height and width as \
                    ``image``

        '''
        to_transform = np.array(image)
        return self._apply_mask(image, mask)


class ImageResizer(ProcessorNode):
    def __init__(self, maintain_ratio = False):
        self._maintain_ratio = maintain_ratio
        super(ImageResizer, self).__init__()

    def _resize_image(self, image : np.array, new_size) -> np.array:
        height, width = new_size
        if height < 0 or width < 0:
            raise ValueError("One of `width` or `height` is a negative value")
        if self._maintain_ratio:
            image = resize_add_padding(image, height, width)
        else:
            image = cv2.resize(image, (width, height))
        return image
    
    def process(self, image : np.array, new_size) -> np.array:
        '''
        Resizes `image` using the dimensions in `new_size`

        - Arguments:
            - image (np.array): shape of (h, w, 3)
            - new_size (tuple): (new_height, new_width)
        
        - Raises:
            - ValueError:
                - If ``new_height`` or ``new_width`` are negative
        '''
        to_transform = np.array(image)
        return self._resize_image(image, new_size)