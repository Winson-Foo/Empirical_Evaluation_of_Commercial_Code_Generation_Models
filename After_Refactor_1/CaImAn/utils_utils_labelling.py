#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Thu Oct 20 15:18:21 2016
"""

import logging
from scipy.ndimage import filters as ft
import caiman as cm

def pre_preprocess_movie_labeling(dview, file_names, median_filter_size=(2, 1, 1),
                                  resize_factors=[.2, .1666666666], diameter_bilateral_blur=4):
    """
    Preprocesses movie labeling data.

    Parameters:
    - dview: The distributed view for parallel computing.
    - file_names: List of file names to be processed.
    - median_filter_size: Tuple of filter sizes for median filtering.
    - resize_factors: List of resize factors for resizing the movie.
    - diameter_bilateral_blur: Diameter for bilateral blur.

    Returns:
    - List of results.
    """

    def pre_process_handle(args):
        """
        Handles the preprocessing of a single file.

        Parameters:
        - args: List of arguments [file_name, resize_factors, diameter_bilateral_blur, median_filter_size].

        Returns:
        - 1 (for now, can be modified to return other useful information).
        """

        fil, resize_factors, diameter_bilateral_blur, median_filter_size = args

        name_log = fil[:-4] + '_LOG'
        logger = logging.getLogger(name_log)
        hdlr = logging.FileHandler(name_log)

        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)

        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)

        logger.info('START')
        logger.info(fil)

        mov = cm.load(fil, fr=30)
        logger.info('Read file')

        mov = mov.resize(1, 1, resize_factors[0])
        logger.info('Resize')

        mov = mov.bilateral_blur_2D(diameter=diameter_bilateral_blur)
        logger.info('Bilateral')

        mov1 = cm.movie(ft.median_filter(mov, median_filter_size), fr=30)
        logger.info('Median filter')

        mov1 = mov1.resize(1, 1, resize_factors[1])
        logger.info('Resize 2')

        mov1 = mov1 - cm.utils.stats.mode_robust(mov1, 0)
        logger.info('Mode')

        mov = mov.resize(1, 1, resize_factors[1])
        logger.info('Resize')

        mov.save(fil[:-4] + '_compress_.tif')
        logger.info('Save 1')

        mov1.save(fil[:-4] + '_BL_compress_.tif')
        logger.info('Save 2')

        return 1

    args = [[name, resize_factors, diameter_bilateral_blur, median_filter_size] for name in file_names]

    if dview is not None:
        file_res = dview.map_sync(pre_process_handle, args)
        dview.results.clear()
    else:
        file_res = list(map(pre_process_handle, args))

    return file_res