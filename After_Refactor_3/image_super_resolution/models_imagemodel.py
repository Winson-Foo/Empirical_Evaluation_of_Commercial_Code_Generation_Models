import numpy as np
from typing import Tuple, Optional, List
from ISR.utils.image_processing import process_array, process_output, split_image_into_overlapping_patches, stich_together


class ImageModel:
    """
    ISR models parent class.

    Contains functions that are common across the super-scaling models.
    """

    def __init__(self, scale: int, model: Any):
        self.scale = scale
        self.model = model

    def process_patches(self, patches: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Processes patches in batches using the network model.

        Args:
            patches: input image patches.
            batch_size: The number of patches processed at a time.

        Returns:
            np.ndarray: output patches
        """
        collect = []
        for i in range(0, len(patches), batch_size):
            batch = self.model.predict(patches[i: i + batch_size])
            if i == 0:
                collect = batch
            else:
                collect = np.append(collect, batch, axis=0)
        return collect

    def process_large_image(
            self,
            input_image_array: np.ndarray,
            by_patch_of_size: Tuple[int, int],
            padding_size: int,
            batch_size: int,
            ) -> np.ndarray:
        """
        Processes a large image by splitting it into patches, processing them with the network model,
        then stitching the patches back together to generate the final image.

        Args:
            input_image_array: input image array.
            by_patch_of_size: the size of a patch.
            padding_size: the padding between patches.
            batch_size: the number of patches processed at a time.
        
        Returns:
            sr_img: image output.
        """
        
        lr_img = process_array(input_image_array, expand=False)
        patches, patch_shape = split_image_into_overlapping_patches(
            lr_img, patch_size=by_patch_of_size, padding_size=padding_size
        )

        output_patches = self.process_patches(patches, batch_size)

        scaled_image_shape = tuple(np.multiply(input_image_array.shape[0:2], self.scale)) + (3,)
        padded_size_scaled = tuple(np.multiply(patch_shape[0:2], self.scale)) + (3,)
        sr_img = stich_together(
            output_patches,
            padded_image_shape=padded_size_scaled,
            target_shape=scaled_image_shape,
            padding_size=padding_size * self.scale,
        )
        return sr_img

    def process_single_image(self, input_image_array: np.ndarray) -> np.ndarray:
        """
        Processes a single image using the network model.

        Args:
            input_image_array: input image array.

        Returns:
            sr_img: output image.
        """
        lr_img = process_array(input_image_array)
        sr_img = self.model.predict(lr_img)[0]
        sr_img = process_output(sr_img)
        return sr_img

    def predict(
            self,
            input_image_array: np.ndarray,
            by_patch_of_size: Optional[Tuple[int, int]] = None,
            padding_size: int = 2,
            batch_size: int = 10,
            ) -> np.ndarray:
        """
        Processes the image array into a suitable format and transforms the network output in a suitable
        image format.

        Args:
            input_image_array: input image array.
            by_patch_of_size: for large image inference. Splits the image into patches of the given size.
            padding_size: for large image inference. Padding between the patches. Increase the value if there is seamlines.
            batch_size: for large image inference. Number of patches processed at a time. Keep low and increase by_patch_of_size instead.

        Returns:
            sr_img: image output.
        """
        if by_patch_of_size:
            sr_img = self.process_large_image(input_image_array, by_patch_of_size, padding_size, batch_size)
        else:
            sr_img = self.process_single_image(input_image_array)

        return sr_img