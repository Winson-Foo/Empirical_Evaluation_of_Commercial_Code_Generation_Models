#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 18:30:19 2017
@author: agiovann
"""

import numpy as np
import caiman as cm
import cv2
from time import time


def motion_correction(file_path, num_frames):
    mc = cm.load(file_path)[:num_frames].motion_correct(3, 3)[0]
    return mc


def calculate_optical_flow(mc):
    mapXY = []
    
    for fr, img in enumerate(mc):
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        mapXY.append(magnitude_spectrum)
        
    return mapXY


def calculate_flow(m, templ):
    num_frames = m.shape[0]
    mapX = np.zeros((num_frames, m.shape[1], m.shape[2]), dtype=np.float32)
    mapY = np.zeros((num_frames, m.shape[1], m.shape[2]), dtype=np.float32)
    map_orig_x = np.zeros((m.shape[1], m.shape[2]), dtype=np.float32)
    map_orig_y = np.zeros((m.shape[1], m.shape[2]), dtype=np.float32)
    
    for j in range(m.shape[1]):
        for i in range(m.shape[2]):
            map_orig_x[j, i] = i
            map_orig_y[j, i] = j
    
    for k in range(num_frames):
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


def process_frames(mc, mapX, mapY, dsfactor):
    num_frames = mc.shape[0]
    mapX_res = cm.movie(mapX).resize(1, 1, 1 / dsfactor)
    mapY_res = cm.movie(mapY).resize(1, 1, 1 / dsfactor)
    
    fact = np.max(mc)
    bl = np.min(mc)
    new_ms = np.zeros(mc[:num_frames].shape)
    
    for counter, mm in enumerate(mc[:num_frames]):
        t1 = time()
        new_img = cv2.remap(
            mm, mapX_res[counter], mapY_res[counter], cv2.INTER_CUBIC, None, cv2.BORDER_CONSTANT)
        new_ms[counter] = new_img
        
    return new_ms


def play_motion_correction_comparison(m, new_ms):
    cm.movie(np.array(m - new_ms)).play()


def main():
    dsfactor = 1
    num_frames = 2000
    
    file_path = '/mnt/ceph/users/agiovann/ImagingData/DanGoodwin/Somagcamp-Fish4-z13-100-400crop256.tif'
    
    mc = motion_correction(file_path, num_frames)
    
    mapXY = calculate_optical_flow(mc)
    
    m = mc.resize(1, 1, dsfactor)
    templ = np.median(m, 0)
    
    mapX, mapY = calculate_flow(m, templ)
    
    new_ms = process_frames(mc, mapX, mapY, dsfactor)
    
    play_motion_correction_comparison(m[:num_frames], new_ms[:num_frames])


if __name__ == "__main__":
    main()