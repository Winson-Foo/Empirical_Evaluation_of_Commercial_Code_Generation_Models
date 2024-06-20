#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import List, Tuple
import caiman as cm
from scipy.ndimage import filters as ft

def pre_preprocess_movie_labeling(dview, file_names: List[str], median_filter_size: Tuple[int, int, int]=(2, 1, 1),
                                  resize_factors: List[float]=[.2, .1666666666], diameter_bilateral_blur: int=4) -> List[int]:
    def pre_process_handle(args):
        fil, resize_factors, diameter_bilateral_blur, median_filter_size = args

        logger = setup_logger(fil[:-4])

        logger.info('START')
        logger.info(fil)

        mov = cm.load(fil, fr=30)
        logger.info('Read file')

        mov = resize_movie(mov, resize_factors[0])
        logger.info('Resize')

        mov = apply_bilateral_blur(mov, diameter_bilateral_blur)
        logger.info('Bilateral blur')

        mov1 = apply_median_filter(mov, median_filter_size)
        logger.info('Median filter')

        mov1 = resize_movie(mov1, resize_factors[1])
        logger.info('Resize 2')

        mov1 = subtract_mode_robust(mov1)
        logger.info('Mode')

        save_movie(mov, fil[:-4] + '_compress_.tif')
        logger.info('Save 1')

        save_movie(mov1, fil[:-4] + '_BL_compress_.tif')
        logger.info('Save 2')
        
        return 1

    args = []
    for name in file_names:
        args.append(
            [name, resize_factors, diameter_bilateral_blur, median_filter_size])

    if dview is not None:
        file_res = dview.map_sync(pre_process_handle, args)
        dview.results.clear()
    else:
        file_res = list(map(pre_process_handle, args))

    return file_res


def setup_logger(logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    hdlr = logging.FileHandler(logger_name + '_LOG')

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)

    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    
    return logger


def resize_movie(movie: cm.movie, resize_factor: float) -> cm.movie:
    return movie.resize(1, 1, resize_factor)


def apply_bilateral_blur(movie: cm.movie, diameter: int) -> cm.movie:
    return movie.bilateral_blur_2D(diameter=diameter)


def apply_median_filter(movie: cm.movie, filter_size: Tuple[int, int, int]) -> cm.movie:
    return cm.movie(ft.median_filter(movie, filter_size), fr=30)


def subtract_mode_robust(movie: cm.movie) -> cm.movie:
    return movie - cm.utils.stats.mode_robust(movie, 0)


def save_movie(movie: cm.movie, filename: str) -> None:
    movie.save(filename)