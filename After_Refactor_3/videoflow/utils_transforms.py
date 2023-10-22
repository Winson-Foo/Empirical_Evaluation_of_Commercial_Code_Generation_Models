from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import cv2

def resize_add_padding(im, target_height, target_width):
    '''
    Resizes an image to a target size, adding padding if necessary to maintain
    the aspect ratio
    - Arguments:
        - im (np.array): shape (h, w, 3)
        - target_height (int): target height
        - target_width (int): target width
    '''
    min_idx = [target_height, target_width].index(min(target_height, target_width))
    ratio = [target_height, target_width][min_idx] / im.shape[min_idx]
    resized_im = cv2.resize(im, (int(im.shape[1] * ratio), int(im.shape[0] * ratio)))

    padded_im = np.zeros((target_height, target_width, 3), dtype=im.dtype)
    padded_im[:resized_im.shape[0], :resized_im.shape[1], :] = resized_im

    return padded_im