from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import cv2
import numpy as np


def resize_add_padding(im, t_h, t_w):
    '''
    Resizes an image to a target size, adding padding if necessary to maintain
    the aspect ratio
    - Arguments:
        - im (np.array): shape (h, w, 3)
        - t_h (int): target height
        - t_w (int): target width
    '''
    # identify the minimal length between height and width
    min_idx = [t_h, t_w].index(min(t_h, t_w))
    
    # determine the scale ratio
    ratio = [t_h, t_w][min_idx] / im.shape[min_idx]
    
    # create a new numpy array with desired shape and datatype
    new_im = np.zeros((t_h, t_w, 3), dtype=im.dtype)
    
    # determine the resized image height and width
    res_h, res_w = int(im.shape[0] * ratio), int(im.shape[1] * ratio)
    
    # resize the image and update the output image
    res_im = cv2.resize(im, (res_w, res_h))
    new_im[:res_h, :res_w, :] = res_im
    
    return new_im