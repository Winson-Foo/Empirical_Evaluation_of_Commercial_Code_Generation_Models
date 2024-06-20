#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Module to filter an image based on RGB values.
"""

import numpy as np
import cv2

def rgb_filter(image_obj, need_rgb):
    """
    Filter an image based on RBG values.

    Args:
        image_obj: The image object to filter.
        need_rgb: The RGB values to filter.

    Returns:
        The filtered image.

    """
    color_offset = 15
    low_rgb = np.array([i - color_offset for i in need_rgb])
    high_rgb = np.array([i + color_offset for i in need_rgb])
    mask = cv2.inRange(image_obj, lowerb=low_rgb, upperb=high_rgb)
    mask = cv2.bitwise_not(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return mask


if __name__ == '__main__':
    pass