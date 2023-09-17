import logging
import os
import re
from typing import Dict, List, Tuple


def get_caiman_data_dir() -> str:
    """Returns the caiman data directory."""
    return os.environ.get("CAIMAN_DATA", os.path.join(os.path.expanduser("~"), "caiman_data"))


def is_caiman_data_dir_exists() -> bool:
    """Checks if the caiman data directory exists."""
    return os.path.isdir(get_caiman_data_dir())


def get_temp_dir() -> str:
    """Returns the caiman temporary directory."""
    caiman_temp = os.environ.get("CAIMAN_TEMP")
    if caiman_temp and os.path.isdir(caiman_temp):
        return caiman_temp

    temp_under_data = os.path.join(get_caiman_data_dir(), "temp")
    if not os.path.isdir(temp_under_data):
        os.makedirs(temp_under_data)
    return temp_under_data


def get_relocated_filename(filename: str) -> str:
    """Returns the absolute path of the file under the temporary directory if no path elements exist in the filename."""
    if "CAIMAN_NEW_TEMPFILE" not in os.environ:
        return filename
    if os.path.basename(filename) == filename:
        return os.path.join(get_temp_dir(), filename)
    return filename


def get_memmap_frames_filename(basename: str, dims: Tuple, frames: int, order: str = 'F') -> str:
    """Returns the filename for memmap frames."""
    d1, d2 = dims[:2]
    d3 = dims[2] if len(dims) == 3 else 1
    return f"{basename}_d1_{d1}_d2_{d2}_d3_{d3}_order_{order}_frames_{frames}.mmap"


def get_derived_presuffix(basename: str, addition: str, swapsuffix: str = None) -> str:
    """Returns the derived filename with an addition and optional suffix swap."""
    fn_base, fn_ext = os.path.splitext(basename)
    if not addition.startswith('_') and not basename.endswith('_'):
        addition = '_' + addition
    if swapsuffix is not None:
        if not swapsuffix.startswith('.'):
            swapsuffix = '.' + swapsuffix
        return fn_base + addition + swapsuffix
    return fn_base + addition + fn_ext


def decode_mmap_filename_dict(basename: str) -> Dict:
    """Decodes the memmap filename and returns a dict with the extracted information."""
    _, fn = os.path.split(basename)
    fn_base, _ = os.path.splitext(fn)
    fpart = fn_base.split('_')[1:]

    ret = {}
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
        print(f"The value of 'T' {ret['T']} differs from 'frames' {ret['frames']}")
    if 'T' not in ret and 'frames' in ret:
        ret['T'] = ret['frames']
    
    return ret


def generate_tot_filename(base_name: str, dims: List[int], order: str) -> str:
    """Generates a 'fname_tot' style filename based on basename, dims, and order."""
    d1, d2, d3 = dims[0], dims[1], dims[2] if len(dims) == 3 else 1
    ret = '_'.join([base_name, 'd1', str(d1), 'd2', str(d2), 'd3', str(d3), 'order', order])
    ret = re.sub(r'(_)+', '_', ret)
    return ret