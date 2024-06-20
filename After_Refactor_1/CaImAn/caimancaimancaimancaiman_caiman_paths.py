import logging
import os
import re
from typing import Dict, List, Tuple


def caiman_datadir() -> str:
    if "CAIMAN_DATA" in os.environ:
        return os.environ["CAIMAN_DATA"]
    else:
        return os.path.join(os.path.expanduser("~"), "caiman_data")


def caiman_datadir_exists() -> bool:
    return os.path.isdir(caiman_datadir())


def get_tempdir() -> str:
    if 'CAIMAN_TEMP' in os.environ:
        if os.path.isdir(os.environ['CAIMAN_TEMP']):
            return os.environ['CAIMAN_TEMP']
        else:
            logging.warning(f"CAIMAN_TEMP is set to nonexistent directory {os.environment['CAIMAN_TEMP']}. Ignoring")
    temp_under_data = os.path.join(caiman_datadir(), "temp")
    if not os.path.isdir(temp_under_data):
        logging.warning(f"Default temporary dir {temp_under_data} does not exist, creating")
        os.makedirs(temp_under_data)
    return temp_under_data


def fn_relocated(fn: str) -> str:
    if not 'CAIMAN_NEW_TEMPFILE' in os.environ:
        return fn
    if str(os.path.basename(fn)) == str(fn):
        return os.path.join(get_tempdir(), fn)
    else:
        return fn


def memmap_frames_filename(basename: str, dims: Tuple, frames: int, order: str = 'F') -> str:
    dimfield_0 = dims[0]
    dimfield_1 = dims[1]
    if len(dims) == 3:
        dimfield_2 = dims[2]
    else:
        dimfield_2 = 1
    return f"{basename}_d1_{dimfield_0}_d2_{dimfield_1}_d3_{dimfield_2}_order_{order}_frames_{frames}.mmap"


def fname_derived_presuffix(basename: str, addition: str, swapsuffix: str = None) -> str:
    fn_base, fn_ext = os.path.splitext(basename)
    if not addition.startswith('_') and not basename.endswith('_'):
        addition = '_' + addition
    if swapsuffix is not None:
        if not swapsuffix.startswith('.'):
            swapsuffix = '.' + swapsuffix
        return fn_base + addition + swapsuffix
    else:
        return fn_base + addition + fn_ext


def decode_mmap_filename_dict(basename: str) -> Dict:
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
    if fpart[-1] != '':
        ret['T'] = int(fpart[-1])
    if 'T' in ret and 'frames' in ret and ret['T'] != ret['frames']:
        print(f"D: The value of 'T' {ret['T']} differs from 'frames' {ret['frames']}")
    if 'T' not in ret and 'frames' in ret:
        ret['T'] = ret['frames']
    return ret


def generate_fname_tot(base_name: str, dims: List[int], order: str) -> str:
    if len(dims) == 2:
        d1, d2, d3 = dims[0], dims[1], 1
    else:
        d1, d2, d3 = dims[0], dims[1], dims[2]
    ret = '_'.join([base_name, 'd1', str(d1), 'd2', str(d2), 'd3', str(d3), 'order', order])
    ret = re.sub(r'(_)+', '_', ret)
    return ret