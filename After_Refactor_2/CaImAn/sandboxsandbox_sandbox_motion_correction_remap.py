#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 18:30:19 2017

@author: agiovann
"""

import numpy as np
import cv2
from time import time


def load_images(file_path):
    return cm.load(file_path)[:2000]


def motion_correction(images):
    return images.motion_correct(3, 3)[0]


def calculate_magnitude_spectrum(images):
    spctr = []
    for fr, img in enumerate(images):
        print(fr)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        spctr.append(magnitude_spectrum)
    return spctr


def build_image_mapping(images, dsfactor):
    inputImage = images[10]
    num_frames = images.shape[0]
    mapX = np.zeros((num_frames, inputImage.shape[0], inputImage.shape[1]), dtype=np.float32)
    mapY = np.zeros((num_frames, inputImage.shape[0], inputImage.shape[1]), dtype=np.float32)
    templ = np.median(images, 0)
    map_orig_x = np.zeros((inputImage.shape[0], inputImage.shape[1]), dtype=np.float32)
    map_orig_y = np.zeros((inputImage.shape[0], inputImage.shape[1]), dtype=np.float32)

    for j in range(inputImage.shape[0]):
        print(j)
        for i in range(inputImage.shape[1]):
            map_orig_x[j, i] = i
            map_orig_y[j, i] = j

    for k in range(num_frames):
        print(k)
        pyr_scale = .5
        levels = 3
        winsize = 20
        iterations = 15
        poly_n = 7
        poly_sigma = 1.2 / 5
        flags = 0
        flow = cv2.calcOpticalFlowFarneback(
            templ, images[k], None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
        mapX[k, :] = map_orig_x + flow[:, :, 0]
        mapY[k, :] = map_orig_y + flow[:, :, 1]
    
    return mapX, mapY


def resize_images(images, dsfactor):
    return images.resize(1, 1, dsfactor)


def apply_image_mapping(images, mapX, mapY):
    num_frames = images.shape[0]
    mapX_res = cm.movie(mapX).resize(1, 1, 1 / dsfactor)
    mapY_res = cm.movie(mapY).resize(1, 1, 1 / dsfactor)
    fact = np.max(images)
    bl = np.min(images)
    times = []
    new_ms = np.zeros(images[:num_frames].shape)
    for counter, mm in enumerate(images[:num_frames]):
        print(counter)
        t1 = time()
        new_img = cv2.remap(
            mm, mapX_res[counter], mapY_res[counter], cv2.INTER_CUBIC, None, cv2.BORDER_CONSTANT)
        new_ms[counter] = new_img
        times.append(time() - t1)
    return new_ms


def subtract_images(original_images, new_images):
    return np.array(original_images - new_images)


def play_movie(images):
    cm.movie(images).play(gain=50., magnification=1)


def concatenate_images(images1, images2):
    return cm.concatenate([cm.movie(np.array(images2), fr=images1.fr), images1],
                          axis=2).resize(1, 1, .5)


def display_images(images):
    pl.subplot(1, 3, 1)
    pl.imshow(np.mean(images[:2000], 0), cmap='gray', vmax=200)
    pl.subplot(1, 3, 2)
    pl.imshow(np.mean(new_ms[:2000], 0), cmap='gray', vmax=200)
    pl.subplot(1, 3, 3)
    pl.imshow(np.mean(new_ms[:2000], 0) - np.mean(images[:2000], 0), cmap='gray')


def calculate_sum(X):
    import numpy as np
    return np.sum(range(X))


if __name__ == "__main__":
    images = load_images('/mnt/ceph/users/agiovann/ImagingData/DanGoodwin/Somagcamp-Fish4-z13-100-400crop256.tif')
    mc = motion_correction(images)
    magnitude_spectrum = calculate_magnitude_spectrum(images)
    mapX, mapY = build_image_mapping(mc, 1)
    resized_images = resize_images(mc, 1)
    new_ms = apply_image_mapping(resized_images, mapX, mapY)
    subtracted_images = subtract_images(mc[:num_frames], new_ms[:num_frames])
    play_movie(subtracted_images)
    concatenated_images = concatenate_images(new_ms[:num_frames], mc[:num_frames])
    display_images(concatenated_images)
    res = pl.map(calculate_sum, range(100000) * 5)