import numpy as np


IMAGE_PADDING_SIZE = 2
PATCH_PADDING_SIZE = 4


def process_image(image_array: np.ndarray, expand: bool=True) -> np.ndarray:
    """Scales a 3-dimensional array into a 4-dimensional batch of size 1."""
    image_batch = image_array / 255.0
    if expand:
        image_batch = np.expand_dims(image_batch, axis=0)
    return image_batch


def process_output(output_tensor: np.ndarray) -> np.ndarray:
    """Transforms a 4-dimensional output tensor into a suitable image format"""
    sr_img = output_tensor.clip(0, 1) * 255
    sr_img = np.uint8(sr_img)
    return sr_img


def pad_patch(image_patch: np.ndarray, padding_size: int, channel_last: bool=True) -> np.ndarray:
    """Pads an image_patch with padding_size edge values."""
    padding = ((padding_size, padding_size), (padding_size, padding_size), (0, 0)) if channel_last else ((0, 0), (padding_size, padding_size), (padding_size, padding_size))
    return np.pad(image_patch, padding, 'edge')


def unpad_patches(image_patches: np.ndarray, padding_size: int) -> np.ndarray:
    """Unpads image_patches by removing edge pixels of size padding_size."""
    return image_patches[:, padding_size:-padding_size, padding_size:-padding_size, :]


def split_image(image_array: np.ndarray, patch_size: int) -> tuple:
    """Splits the image into partially overlapping patches."""
    x_remainder = image_array.shape[0] % patch_size
    y_remainder = image_array.shape[1] % patch_size

    # Add padding to make the image divisible by patch size
    x_extend = (patch_size - x_remainder) % patch_size
    y_extend = (patch_size - y_remainder) % patch_size
    extended_image = np.pad(image_array, ((0, x_extend), (0, y_extend), (0, 0)), 'edge')

    # Add padding around the image to simplify computations
    padded_image = pad_patch(extended_image, PATCH_PADDING_SIZE, channel_last=True)

    xmax, ymax, _ = padded_image.shape
    x_lefts = range(PATCH_PADDING_SIZE, xmax - PATCH_PADDING_SIZE, patch_size)
    y_tops = range(PATCH_PADDING_SIZE, ymax - PATCH_PADDING_SIZE, patch_size)
    patches = [padded_image[x_left-PATCH_PADDING_SIZE:x_left+patch_size+PATCH_PADDING_SIZE, y_top-PATCH_PADDING_SIZE:y_top+patch_size+PATCH_PADDING_SIZE, :] for x_left in x_lefts for y_top in y_tops]
    return np.array(patches), padded_image.shape


def stich_together(patches: np.ndarray, padded_image_shape: tuple, target_shape: tuple) -> np.ndarray:
    """Reconstructs the image from overlapping patches."""
    xmax, ymax, _ = padded_image_shape
    patches = unpad_patches(patches, PATCH_PADDING_SIZE)
    patch_size = patches.shape[1]
    n_patches_per_row = ymax // patch_size
    complete_image = np.zeros((xmax, ymax, 3))
    row = -1
    col = 0

    for i, patch in enumerate(patches):
        if i % n_patches_per_row == 0:
            row += 1
            col = 0
        complete_image[row*patch_size: (row+1)*patch_size, col*patch_size: (col+1)*patch_size] = patch
        col += 1

    return complete_image[:target_shape[0], :target_shape[1], :]