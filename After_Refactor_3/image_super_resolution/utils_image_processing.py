import numpy as np


def process_array(image_array, expand=True):
    """
    Scales a 3-dimensional array into a 4-dimensional batch of size 1.
    
    Args:
        image_array: A numpy array of the input image.
        expand: A bool indicating whether to expand the dimensions.
        
    Returns:
        A numpy array of the scaled image.
    """
    
    scaled_array = image_array / 255.0
    if expand:
        scaled_array = np.expand_dims(scaled_array, axis=0)
    return scaled_array


def process_output(output_tensor):
    """
    Transforms a 4-dimensional output tensor into a suitable image format.
    
    Args:
        output_tensor: A 4-dimensional output tensor.
        
    Returns:
        A numpy array of the transformed image.
    """
    
    clipped_tensor = np.clip(output_tensor, 0, 1) * 255
    transformed_array = np.uint8(clipped_tensor)
    return transformed_array


def pad_patch(image_patch, padding_size, channel_last=True):
    """
    Pads an image_patch with edge values to a new size.

    Args:
        image_patch: A numpy array of the image patch.
        padding_size: An int indicating the size of the overlapping area.
        channel_last: A bool indicating whether the last axis is a channel axis.
        
    Returns:
        A numpy array of the padded image patch.
    """
    
    if channel_last:
        return np.pad(
            image_patch,
            ((padding_size, padding_size), (padding_size, padding_size), (0, 0)),
            'edge',
        )
    else:
        return np.pad(
            image_patch,
            ((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
            'edge',
        )


def unpad_patches(image_patches, padding_size):
    """
    Removes padding from image_patches.

    Args:
        image_patches: A numpy array of image patches.
        padding_size: An int indicating the size of the overlapping area.
        
    Returns:
        A numpy array of the unpadded image patches.
    """
    
    return image_patches[:, padding_size:-padding_size, padding_size:-padding_size, :]


def split_image_into_overlapping_patches(image_array, patch_size, padding_size=2):
    """
    Splits an image into partially overlapping patches with a given patch size and padding size.

    Padding is added to the image to make it divisible by the patch_size.

    Args:
        image_array: A numpy array of the input image.
        patch_size: An int indicating the size of patches from the original image.
        padding_size: An int indicating the size of the overlapping area.

    Returns:
        A tuple containing a numpy array of the image patches and the shape of the padded image.
    """
    
    x_max, y_max, _ = image_array.shape
    x_remainder = x_max % patch_size
    y_remainder = y_max % patch_size

    x_extend = (patch_size - x_remainder) % patch_size
    y_extend = (patch_size - y_remainder) % patch_size
    
    extended_image = np.pad(image_array, ((0, x_extend), (0, y_extend), (0, 0)), 'edge')
    
    padded_image = pad_patch(extended_image, padding_size, channel_last=True)

    x_max, y_max, _ = padded_image.shape
    patches = []

    x_lefts = range(padding_size, x_max - padding_size, patch_size)
    y_tops = range(padding_size, y_max - padding_size, patch_size)

    for x in x_lefts:
        for y in y_tops:
            x_left = x - padding_size
            y_top = y - padding_size
            x_right = x + patch_size + padding_size
            y_bottom = y + patch_size + padding_size
            patch = padded_image[x_left:x_right, y_top:y_bottom, :]
            patches.append(patch)

    return np.array(patches), padded_image.shape


def stich_together(patches, padded_image_shape, target_shape, padding_size=4):
    """
    Reconstructs the image from overlapping patches.

    Args:
        patches: A numpy array of the image patches obtained with `split_image_into_overlapping_patches`.
        padded_image_shape: A tuple containing the shape of the padded image.
        target_shape: A tuple containing the shape of the final image.
        padding_size: An int indicating the size of the overlapping area.

    Returns:
        A numpy array of the reconstructed image.
    """
    
    x_max, y_max, _ = padded_image_shape
    patches = unpad_patches(patches, padding_size)
    patch_size = patches.shape[1]
    n_patches_per_row = y_max // patch_size

    output_image = np.zeros((x_max, y_max, 3))

    row = -1
    col = 0
    for i in range(len(patches)):
        if i % n_patches_per_row == 0:
            row += 1
            col = 0
        output_image[
        row * patch_size: (row + 1) * patch_size, col * patch_size: (col + 1) * patch_size, :
        ] = patches[i]
        col += 1
    return output_image[0: target_shape[0], 0: target_shape[1], :]