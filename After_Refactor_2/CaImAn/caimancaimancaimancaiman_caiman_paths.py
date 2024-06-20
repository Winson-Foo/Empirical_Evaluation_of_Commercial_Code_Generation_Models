#!/usr/bin/env python
"""Utilities for retrieving paths around the caiman package and its datadirs"""

import logging
import os
import re
from typing import Dict, List, Tuple


def caiman_datadir() -> str:
    """
    Return the path to the caiman data directory.
    The datadir is a user-configurable place that holds a user-modifiable copy of
    data that the Caiman libraries need to function, alongside code demos and other things.
    This is meant to be separate from the library install of Caiman, which may be installed
    into the global python library path (or into a conda path or somewhere else messy).
    """
    caiman_data_env = os.environ.get("CAIMAN_DATA")
    if caiman_data_env:
        return caiman_data_env
    else:
        return os.path.join(os.path.expanduser("~"), "caiman_data")


def caiman_datadir_exists() -> bool:
    """
    Check if the caiman datadir exists.
    """
    return os.path.isdir(caiman_datadir())


def get_tempdir() -> str:
    """
    Return the temporary directory where CaImAn can store files.
    Controlled mainly by environment variables.
    """
    caiman_temp_env = os.environ.get("CAIMAN_TEMP")

    if caiman_temp_env and os.path.isdir(caiman_temp_env):
        return caiman_temp_env

    temp_under_data = os.path.join(caiman_datadir(), "temp")

    if not os.path.isdir(temp_under_data):
        logging.warning(f"Default temporary dir {temp_under_data} does not exist, creating")
        os.makedirs(temp_under_data)

    return temp_under_data


def fn_relocated(fn: str) -> str:
    """
    Return the absolute pathname of a file located in the caiman temp directory.
    If the provided filename does not contain any path elements, it will be moved to the temp directory.
    """
    if os.environ.get("CAIMAN_NEW_TEMPFILE"):
        return os.path.join(get_tempdir(), fn)

    if os.path.basename(fn) == fn:
        return os.path.join(get_tempdir(), fn)
    else:
        return fn


def memmap_frames_filename(basename: str, dims: Tuple, frames: int, order: str = 'F') -> str:
    """
    Return the filename for the memmap frames file based on the provided parameters.
    """
    d1, d2, d3 = dims[0], dims[1], dims[2] if len(dims) == 3 else 1
    return f"{basename}_d1_{d1}_d2_{d2}_d3_{d3}_order_{order}_frames_{frames}.mmap"


def fname_derived_presuffix(basename: str, addition: str, swapsuffix: str = None) -> str:
    """
    Return the filename with an extension modified by adding the provided addition.
    """
    fn_base, fn_ext = os.path.splitext(basename)

    if not addition.startswith('_') and not basename.endswith('_'):
        addition = '_' + addition

    if swapsuffix:
        if not swapsuffix.startswith('.'):
            swapsuffix = '.' + swapsuffix

        return fn_base + addition + swapsuffix
    else:
        return fn_base + addition + fn_ext


def decode_mmap_filename_dict(basename: str) -> Dict:
    """
    Decode a memmap filename and return a dictionary with encoded information.
    """
    ret = {}
    _, fn = os.path.split(basename)
    fn_base, _ = os.path.splitext(fn)
    fpart = fn_base.split('_')[1:]

    for field in ['d1', 'd2', 'd3', 'order', 'frames']:
        for i in range(len(fpart) - 1, -1, -1):
            if field == fpart[i]:
                if field == 'order':
                    ret[field] = fpart[i + 1]
                else:
                    ret[field] = int(fpart[i + 1])

    if fpart[-1]:
        ret['T'] = int(fpart[-1])

    if 'T' in ret and 'frames' in ret and ret['T'] != ret['frames']:
        print(f"The value of 'T' {ret['T']} differs from 'frames' {ret['frames']}")

    if 'T' not in ret and 'frames' in ret:
        ret['T'] = ret['frames']

    return ret


def generate_fname_tot(base_name: str, dims: List[int], order: str) -> str:
    """
    Generate a "fname_tot" style filename based on the provided parameters.
    """
    d1, d2, d3 = dims[0], dims[1], dims[2] if len(dims) == 3 else 1
    ret = '_'.join([base_name, 'd1', str(d1), 'd2', str(d2), 'd3', str(d3), 'order', order])
    ret = re.sub(r'(_)+', '_', ret)
    return ret