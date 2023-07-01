from pathlib import Path
import numpy as np
from nose.tools import assert_equal

from caiman.mmapping import load_memmap
from caiman.paths import caiman_datadir


def get_file_path(file_name):
    """
    Returns the full file path given the file name.
    """
    return Path(caiman_datadir()) / "testdata" / file_name


def test_load_raises_wrong_ext():
    """
    Test for load_memmap function when given file has incorrect extension.
    It should raise a ValueError.
    """
    file_name = "a.mmapp"
    try:
        load_memmap(file_name)
    except ValueError:
        assert True, "ValueError not raised"
    else:
        assert False, "ValueError not raised"


def test_load_raises_multiple_ext():
    """
    Test for load_memmap function when given file has multiple extensions.
    It should raise a ValueError.
    """
    file_name = "a.mmap.mma"
    try:
        load_memmap(file_name)
    except ValueError:
        assert True, "ValueError not raised"
    else:
        assert False, "ValueError not raised"


def setup_2d_mmap():
    """
    Set up function for creating a 2D memmap file for testing.
    """
    file_path = get_file_path("memmap__d1_10_d2_11_d3_1_order_F_frames_12_.mmap")
    np.memmap(file_path, mode="w+", dtype=np.float32, shape=(12, 10, 11, 13), order="F")


def teardown_2d_mmap():
    """
    Teardown function for removing the 2D memmap file after testing.
    """
    file_path = get_file_path("memmap__d1_10_d2_11_d3_1_order_F_frames_12_.mmap")
    file_path.unlink()


def setup_3d_mmap():
    """
    Set up function for creating a 3D memmap file for testing.
    """
    file_path = get_file_path("memmap__d1_10_d2_11_d3_13_order_F_frames_12_.mmap")
    np.memmap(file_path, mode="w+", dtype=np.float32, shape=(12, 10, 11, 13), order="F")


def teardown_3d_mmap():
    """
    Teardown function for removing the 3D memmap file after testing.
    """
    file_path = get_file_path("memmap__d1_10_d2_11_d3_13_order_F_frames_12_.mmap")
    file_path.unlink()


def test_load_successful_2d():
    """
    Test for successful loading of a 2D memmap file using load_memmap function.
    """
    file_path = get_file_path("memmap__d1_10_d2_11_d3_1_order_F_frames_12_.mmap")
    Yr, (d1, d2), T = load_memmap(str(file_path))
    assert_equal((d1, d2), (10, 11), "Wrong shape of 2D memmap")
    assert_equal(T, 12, "Wrong number of frames")
    assert isinstance(Yr, np.memmap), "Yr is not a memmap object"


def test_load_successful_3d():
    """
    Test for successful loading of a 3D memmap file using load_memmap function.
    """
    file_path = get_file_path("memmap__d1_10_d2_11_d3_13_order_F_frames_12_.mmap")
    Yr, (d1, d2, d3), T = load_memmap(str(file_path))
    assert_equal((d1, d2, d3), (10, 11, 13), "Wrong shape of 3D memmap")
    assert_equal(T, 12, "Wrong number of frames")
    assert isinstance(Yr, np.memmap), "Yr is not a memmap object"