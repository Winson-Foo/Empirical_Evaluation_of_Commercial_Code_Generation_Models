#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import numpy as np
import cv2


def apply_rgb_filter(image: np.array, rgb_values: list[int]) -> np.array:
    """
    Apply an RGB filter to an image, keeping only the pixels within the specified RGB value range.
    :param image: The input image as a numpy array
    :param rgb_values: The RGB values as a list of integers [R, G, B]
    :return: The filtered image as a numpy array
    """
    lower_rgb = np.array([i - 15 for i in rgb_values])
    upper_rgb = np.array([i + 15 for i in rgb_values])

    mask = cv2.inRange(image, lowerb=lower_rgb, upperb=upper_rgb)
    mask = cv2.bitwise_not(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    return mask


if __name__ == '__main__':
    pass