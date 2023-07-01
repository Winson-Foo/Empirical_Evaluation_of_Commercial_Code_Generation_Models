#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import io
import cv2
import PIL.Image as Pil_Image
import numpy as np


def coord_calc(param, is_range=True, is_integer=True):

    result_group = []
    start_h = param['start_pos'][1]
    end_h = start_h + param['corp_size'][1]
    for row in range(param['corp_num'][1]):
        start_w = param['start_pos'][0]
        end_w = start_w + param['corp_size'][0]
        for col in range(param['corp_num'][0]):
            pos_range = [[start_w, end_w], [start_h, end_h]]
            t = lambda x: int(x) if is_integer else x
            pos_center = [t((start_w + end_w)/2), t((start_h + end_h)/2)]
            result_group.append(pos_range if is_range else pos_center)
            start_w = end_w + param['interval_size'][0]
            end_w = start_w + param['corp_size'][0]
        start_h = end_h + param['interval_size'][1]
        end_h = start_h + param['corp_size'][1]
    return result_group


def parse_multi_img(image_bytes, param_group):
    img_bytes = image_bytes[0]
    image_arr = np.array(Pil_Image.open(io.BytesIO(img_bytes)).convert('RGB'))
    if len(image_arr.shape) == 3:
        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
    # image_arr = np.fromstring(img_bytes, np.uint8)
    # print(image_arr.shape)
    image_arr = image_arr.swapaxes(0, 1)
    group = []
    for p in param_group:
        pos_ranges = coord_calc(p, True, True)
        for pos_range in pos_ranges:
            corp_arr = image_arr[pos_range[0][0]: pos_range[0][1], pos_range[1][0]: pos_range[1][1]]
            corp_arr = cv2.imencode('.png', np.swapaxes(corp_arr, 0, 1))[1]
            corp_bytes = bytes(bytearray(corp_arr))
            group.append(corp_bytes)
    return group


def get_coordinate(label: str, param_group, title_index=None):
    if title_index is None:
        title_index = [0]
    param = param_group[-1]
    coord_map = coord_calc(param, is_range=False, is_integer=True)
    index_group = get_pair_index(label=label, title_index=title_index)
    return [coord_map[i] for i in index_group]


def get_pair_index(label: str, title_index=None):
    if title_index is None:
        title_index = [0]
    max_index = max(title_index)

    label_group = label.split(',')
    titles = [label_group[i] for i in title_index]

    index_group = []
    for title in titles:
        for i, item in enumerate(label_group[max_index+1:]):
            if item == title:
                index_group.append(i)
        index_group = [i for i in index_group]
    return index_group


if __name__ == '__main__':
    import os
    import hashlib
    root_dir = r"H:\Task\Trains\d111_Trains"
    target_dir = r"F:\1q2"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    _param_group = [
        {
          "start_pos": [20, 50],
          "interval_size": [20, 20],
          "corp_num": [4, 2],
          "corp_size": [60, 60]
        }
    ]
    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        with open(path, "rb") as f:
            file_bytes = [f.read()]
        group = parse_multi_img(file_bytes, _param_group)
        for b in group:
            tag = hashlib.md5(b).hexdigest()
            p = os.path.join(target_dir, "{}.png".format(tag))
            with open(p, "wb") as f:
                f.write(b)