#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 18:30:19 2017
"""

import cv2
import numpy as np
import caiman as cm
import pylab as pl
from time import time

def preprocess_data():
    mc = cm.load('/mnt/ceph/users/agiovann/ImagingData/DanGoodwin/Somagcamp-Fish4-z13-100-400crop256.tif')[:2000].motion_correct(3, 3)[0]
    return mc

def calculate_magnitude_spectrum(mc):
    spctr = []
    for fr, img in enumerate(cm.load('/mnt/ceph/users/agiovann/ImagingData/DanGoodwin/Somagcamp-Fish4-z13-100-400crop256.tif')):
        print(fr)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        spctr.append(magnitude_spectrum)
    return spctr

def optical_flow(mc):
    dsfactor = 1
    m = mc.resize(1, 1, dsfactor)
    num_frames = m.shape[0]
    inputImage = m[10]
    mapX = np.zeros((num_frames, inputImage.shape[0], inputImage.shape[1]), dtype=np.float32)
    mapY = np.zeros((num_frames, inputImage.shape[0], inputImage.shape[1]), dtype=np.float32)
    templ = np.median(m, 0)
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
            templ, m[k], None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
        mapX[k, :] = map_orig_x + flow[:, :, 0]
        mapY[k, :] = map_orig_y + flow[:, :, 1]

    return mapX, mapY

def remap_frames(mc, mapX_res, mapY_res):
    num_frames = mc.shape[0]
    mapX_res = cm.movie(mapX).resize(1, 1, 1 / dsfactor)
    mapY_res = cm.movie(mapY).resize(1, 1, 1 / dsfactor)
    fact = np.max(m)
    bl = np.min(m)
    times = []
    new_ms = np.zeros(mc[:num_frames].shape)
    for counter, mm in enumerate(mc[:num_frames]):
        print(counter)
        t1 = time()
        new_img = cv2.remap(
            mm, mapX_res[counter], mapY_res[counter], cv2.INTER_CUBIC, None, cv2.BORDER_CONSTANT)
        new_ms[counter] = new_img
        times.append(time() - t1)
    
    return new_ms

def main():
    mc = preprocess_data()
    spctr = calculate_magnitude_spectrum(mc)
    mapX, mapY = optical_flow(mc)
    new_ms = remap_frames(mc, mapX_res, mapY_res)

    cm.movie(np.array(mc[:num_frames] - new_ms[:num_frames])).play(gain=50., magnification=1)
    cm.concatenate([cm.movie(np.array(new_ms[:num_frames]), fr=mc.fr), mc[:num_frames]], axis=2).resize(1, 1, .5).play(gain=2, magnification=3, fr=30, offset=-100)
    pl.subplot(1, 3, 1)
    pl.imshow(np.mean(m[:2000], 0), cmap='gray', vmax=200)
    pl.subplot(1, 3, 2)
    pl.imshow(np.mean(new_ms[:2000], 0), cmap='gray', vmax=200)
    pl.subplot(1, 3, 3)
    pl.imshow(np.mean(new_ms[:2000], 0) - np.mean(m[:2000], 0), cmap='gray')
    cm.movie(np.array(m[:2000] - new_ms[:2000])).play()

if __name__ == "__main__":
    main()