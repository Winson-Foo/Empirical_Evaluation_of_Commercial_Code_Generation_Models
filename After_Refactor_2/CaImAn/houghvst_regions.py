import numpy as np
from skimage.util import view_as_windows


def select_patches(img: np.ndarray, n_patches: int, patch_size: int = 32, overlapping: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (img.shape[0] - patch_size, img.shape[1] - patch_size)
    indices = np.arange(np.prod(shape))

    if not overlapping:
        starting_points = np.ones(shape, dtype=bool)
        available = np.ones(shape, dtype=bool)

        patch_indices = []
        patches = []
        masks = []
        while len(patches) < n_patches:
            if not np.any(starting_points):
                break

            k = np.random.choice(indices[starting_points.flat])
            k0, k1 = np.unravel_index(k, shape)

            sl0 = slice(k0, np.minimum(k0 + patch_size, shape[0]))
            sl1 = slice(k1, np.minimum(k1 + patch_size, shape[1]))
            if not np.all(available[sl0, sl1]):
                starting_points[k0, k1] = False
                continue

            available[sl0, sl1] = False

            sl0 = slice(np.maximum(k0 - patch_size + 1, 0), np.minimum(k0 + patch_size, shape[0]))
            sl1 = slice(np.maximum(k1 - patch_size + 1, 0), np.minimum(k1 + patch_size, shape[1]))
            starting_points[sl0, sl1] = False

            patch_indices.append((k0, k1))
            patches.append(img[k0:k0 + patch_size, k1:k1 + patch_size])
            m = np.zeros(img.shape, dtype=bool)
            m[k0:k0 + patch_size, k1:k1 + patch_size] = True
            masks.append(m)

        patch_indices = np.array(patch_indices)
        patches = np.array(patches)
        masks = np.array(masks)
        labels = sum([m * (i + 1) for i, m in enumerate(masks)])
        return patch_indices, patches, masks, labels
    else:
        k = np.random.choice(indices, size=n_patches)
        patch_indices = np.vstack(np.unravel_index(k, shape)).T
        patches = [img[k0:k0 + patch_size, k1:k1 + patch_size] for k0, k1 in patch_indices]
        patches = np.array(patches)
        return patch_indices, patches


def image_to_columns(img: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    arr = view_as_windows(img, window_size)
    arr = arr[::stride, ::stride, :, :]
    arr = arr.reshape(np.prod(arr.shape[:-2]), window_size, window_size)
    return arr