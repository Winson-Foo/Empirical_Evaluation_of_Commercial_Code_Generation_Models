#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import io
import cv2
import numpy as np
from itertools import groupby
from PIL import ImageSequence, Image


def split_frames(image_obj, need_frame=None):
    image_seq = ImageSequence.all_frames(image_obj)
    image_arr_last = [np.asarray(image_seq[-1])] if -1 in need_frame and len(need_frame) > 1 else []
    image_arr = [np.asarray(item) for i, item in enumerate(image_seq) if (i in need_frame or need_frame == [-1])]
    image_arr += image_arr_last
    return image_arr


def concat_arr(img_arr):
    if len(img_arr) < 2:
        return img_arr[0]
    all_slice = img_arr[0]
    for im_slice in img_arr[1:]:
        all_slice = np.concatenate((all_slice, im_slice), axis=1)
    return all_slice


def numpy_to_bytes(numpy_arr):
    cv_img = cv2.imencode('.png', numpy_arr)[1]
    img_bytes = bytes(bytearray(cv_img))
    return img_bytes


def concat_frames(image_obj, need_frame=None):
    if not need_frame:
        need_frame = [0]
    img_arr = split_frames(image_obj, need_frame)
    img_arr = concat_arr(img_arr)
    return img_arr


def blend_arr(img_arr):
    if len(img_arr) < 2:
        return img_arr[0]
    all_slice = img_arr[0]
    for im_slice in img_arr[1:]:
        all_slice = cv2.addWeighted(all_slice, 0.5, im_slice, 0.5, 0)
    # all_slice = cv2.equalizeHist(all_slice)
    return all_slice


def blend_frame(image_obj, need_frame=None):
    if not need_frame:
        need_frame = [-1]
    img_arr = split_frames(image_obj, need_frame)
    img_arr = blend_arr(img_arr)
    if len(img_arr.shape) > 2 and img_arr.shape[2] == 3:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    img_arr = cv2.equalizeHist(img_arr)
    return img_arr


def all_frames(image_obj):
    if isinstance(image_obj, list):
        image_obj = image_obj[0]
    stream = io.BytesIO(image_obj)
    pil_image = Image.open(stream)
    image_seq = ImageSequence.all_frames(pil_image)
    array_seq = [np.asarray(im.convert("RGB")) for im in image_seq]
    # [1::2]
    bytes_arr = [cv2.imencode('.png', img_arr)[1] for img_arr in array_seq]
    split_flag = b'\x99\x99\x99\x00\xff\xff999999.........99999\xff\x00\x99\x99\x99'
    return split_flag.join(bytes_arr).split(split_flag)


def get_continuity_max(src: list):
    if not src:
        return ""
    elem_cont_len = lambda x: max(len(list(g)) for k, g in groupby(src) if k == x)
    target_list = [elem_cont_len(i) for i in src]
    target_index = target_list.index(max(target_list))
    return src[target_index]


if __name__ == "__main__":
    pass
