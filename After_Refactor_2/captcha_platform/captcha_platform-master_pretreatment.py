#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import cv2


class ImageProcessor:

    def __init__(self, image):
        self.image = image

    def binarization(self, threshold, modify=False):
        _, binarized_image = cv2.threshold(self.image, threshold, 255, cv2.THRESH_BINARY)
        if modify:
            self.image = binarized_image
        return binarized_image

    def apply_preprocessing(self, binaryzation=-1):
        if binaryzation > 0:
            self.binarization(binaryzation, True)
        return self.image


def preprocess_image(image, binaryzation=-1):
    processor = ImageProcessor(image)
    return processor.apply_preprocessing(binaryzation)


def preprocess_image_by_func(exec_map, key, src_arr):
    if not exec_map:
        return src_arr
    target_arr = cv2.cvtColor(src_arr, cv2.COLOR_RGB2BGR)
    for sentence in exec_map.get(key):
        if sentence.startswith("@@"):
            target_arr = eval(sentence[2:])
        elif sentence.startswith("$$"):
            exec(sentence[2:])
    return cv2.cvtColor(target_arr, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    pass