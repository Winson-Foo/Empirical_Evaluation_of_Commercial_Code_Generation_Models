from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import cv2
import numpy as np


def resize_add_padding(im, target_height, target_width):
    '''
    Resizes an image to a target size, adding padding if necessary to maintain
    the aspect ratio
    - Arguments:
        - im (np.array): shape (h, w, 3)
        - target_height (int): target height
        - target_width (int): target width
    '''
    min_target = min(target_height, target_width)
    min_im = im.shape[min_idx]
    ratio = min_target / min_im
    new_im = np.zeros((target_height, target_width, 3), dtype=im.dtype)
    res_h, res_w = int(im.shape[0] * ratio), int(im.shape[1] * ratio)
    res_im = cv2.resize(im, (res_w, res_h))
    new_im[:res_h, :res_w, :] = res_im
    return new_im