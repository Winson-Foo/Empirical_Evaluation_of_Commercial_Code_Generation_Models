import numpy as np
from ISR.utils.image_processing import process_array, process_output, split_image_into_overlapping_patches, stich_together

PADDING_SIZE = 2
BATCH_SIZE = 10


class ImageModel:
    """ISR models parent class.

    Contains functions that are common across the super-scaling models.
    """
    
    def predict(self, input_image_array, by_patch_of_size=None):
        """
        Processes the image array into a suitable format
        and transforms the network output in a suitable image format.

        Args:
            input_image_array: input image array.
            by_patch_of_size: for large image inference. Splits the image into
                patches of the given size.
        Returns:
            sr_img: image output.
        """
        if by_patch_of_size:
            lr_img = process_array(input_image_array, expand=False)
            patches, patch_shape = split_image_into_overlapping_patches(
                lr_img, patch_size=by_patch_of_size, padding_size=PADDING_SIZE
            )
            batch_output = self._predict_large_image_patches(patches)
            sr_img = self._stitch_large_image_patches(batch_output, input_image_array, patch_shape)
        else:
            lr_img = process_array(input_image_array)
            sr_img = self.model.predict(lr_img)[0]
        
        sr_img = process_output(sr_img)
        return sr_img
    
    def _predict_large_image_patches(self, patches):
        """
        Predicts the output for a large image that has been split into patches.

        Args:
            patches: list of patches.
        Returns:
            batch_output: output of the patches.
        """
        batch_output = []
        for i in range(0, len(patches), BATCH_SIZE):
            batch = self.model.predict(patches[i: i + BATCH_SIZE])
            if i == 0:
                batch_output = batch
            else:
                batch_output = np.append(batch_output, batch, axis=0)
        
        return batch_output
    
    def _stitch_large_image_patches(self, batch_output, original_image, patch_shape):
        """
        Stitches the patches back together to form the large image.

        Args:
            batch_output: output of the patches.
            original_image: original input image.
            patch_shape: shape of the patches.
        Returns:
            sr_img: output image.
        """
        scale = self.scale
        padded_size_scaled = tuple(np.multiply(patch_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(original_image.shape[0:2], scale)) + (3,)
        sr_img = stich_together(
            batch_output,
            padded_image_shape=padded_size_scaled,
            target_shape=scaled_image_shape,
            padding_size=PADDING_SIZE * scale,
        )
        return sr_img