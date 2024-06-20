#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Thu Oct 20 15:18:21 2016

Refactored by [Your Name]

This module provides functions for preprocessing movie labeling data.

"""

import logging
from scipy.ndimage import filters as ft
import caiman as cm


def preprocess_movie_labeling(file_names, resize_factors=(0.20, 0.1666666666), diameter_bilateral_blur=4):
    """
    Preprocesses movie labeling data by resizing, applying bilateral blur, median filtering,
    and subtracting mode.

    Args:
        file_names (list): List of file names to process.
        resize_factors (tuple, optional): Tuple containing resize factors for the two resize operations.
        diameter_bilateral_blur (int, optional): Diameter for the bilateral blur filter.

    Returns:
        list: A list of results indicating the success of each file processing.

    """

    def setup_logger(file_name):
        logger = logging.getLogger(file_name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        file_handler = logging.FileHandler(file_name)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def pre_process(file_name, resize_factors, diameter_bilateral_blur):
        try:
            logger = setup_logger(file_name[:-4] + '_LOG')
            logger.info('START')
            logger.info(file_name)

            mov = cm.load(file_name, fr=30)
            logger.info('Read file')

            mov = mov.resize(1, 1, resize_factors[0])
            logger.info('Resize')

            mov = mov.bilateral_blur_2D(diameter=diameter_bilateral_blur)
            logger.info('Bilateral')

            mov1 = cm.movie(ft.median_filter(mov, (2, 1, 1)), fr=30)
            logger.info('Median filter')

            mov1 = mov1.resize(1, 1, resize_factors[1])
            logger.info('Resize 2')

            mov1 = mov1 - cm.utils.stats.mode_robust(mov1, 0)
            logger.info('Mode')

            mov = mov.resize(1, 1, resize_factors[1])
            logger.info('Resize')

            save_file_name = file_name[:-4] + '_compress.tif'
            mov.save(save_file_name)
            logger.info('Save 1')

            save_file_name = file_name[:-4] + '_BL_compress.tif'
            mov1.save(save_file_name)
            logger.info('Save 2')

            return 1
        except Exception as e:
            logger.error(str(e))
            return 0

    results = []
    for file_name in file_names:
        result = pre_process(file_name, resize_factors, diameter_bilateral_blur)
        results.append(result)

    return results