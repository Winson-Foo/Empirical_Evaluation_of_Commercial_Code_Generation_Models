#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import numpy as np
import cv2


def apply_rgb_filter(image, target_rgb):
    low_rgb = np.array([c-15 for c in target_rgb])
    high_rgb = np.array([c+15 for c in target_rgb])
    filtered = cv2.inRange(image, lowerb=low_rgb, upperb=high_rgb)
    filtered = cv2.bitwise_not(filtered)
    filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
    return filtered


def main():
    # Example usage of apply_rgb_filter
    image = cv2.imread('image.png')
    target_rgb = [255, 0, 0]
    filtered_image = apply_rgb_filter(image, target_rgb)
    cv2.imwrite('filtered_image.png', filtered_image)


if __name__ == '__main__':
    main()