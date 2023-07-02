#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import cv2
import PIL.Image as PilImage
import numpy as np
import onnxruntime as ort
from enum import Enum, unique
from middleware.resource import color_model


@unique
class TargetColor(Enum):
    Red = 1
    Blue = 2
    Yellow = 3
    Black = 4


color_map = {
    'black': TargetColor.Black,
    'red': TargetColor.Red,
    'blue': TargetColor.Blue,
    'yellow': TargetColor.Yellow,
}


class ColorFilter:

    def __init__(self):
        self.model_onnx = color_model
        self.sess = ort.InferenceSession(self.model_onnx)

    def predict_color(self, image_batch: np.ndarray, color: TargetColor) -> list:
        dense_decoded_code = self.sess.run(["dense_decoded:0"], input_feed={
            "input:0": image_batch,
        })
        result = dense_decoded_code[0][0].tolist()
        return [i for i, c in enumerate(result) if c == color.value]


def process_image(image_path: str, target_path: str, color_extract: ColorExtract) -> None:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        result = color_extract.separate_color(image_bytes, color_map['red'])
    with open(target_path, "wb") as f:
        f.write(result)


def batch_process_images(source_dir: str, target_dir: str, color_extract: ColorExtract) -> None:
    import os
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    source_names = os.listdir(source_dir)
    for i, name in enumerate(source_names):
        image_path = os.path.join(source_dir, name)
        target_path = os.path.join(target_dir, name)
        
        if i % 100 == 0:
            print(i)
        
        process_image(image_path, target_path, color_extract)


if __name__ == '__main__':
    pass
    # source_dir = r'E:\***'
    # target_dir = r'E:\***'
    # color_extract = ColorExtract()
    # batch_process_images(source_dir, target_dir, color_extract)