import numpy as np
import pathlib
import nose

from caiman import mmapping
from caiman.paths import caiman_datadir


TWO_D_FNAME = (
    pathlib.Path(caiman_datadir()) / "testdata/memmap__d1_10_d2_11_d3_1_order_F_frames_12_.mmap"
)
THREE_D_FNAME = (
    pathlib.Path(caiman_datadir()) / "testdata/memmap__d1_10_d2_11_d3_13_order_F_frames_12_.mmap"
)


def test_load_raises_wrong_ext():
    """
    Test if loading memmap with wrong extension raises a value error.
    """
    fname = "a.mmapp"
    nose.tools.assert_raises(ValueError, mmapping.load_memmap, fname)


def test_load_raises_multiple_ext():
    """
    Test if loading memmap with multiple extensions raises a value error.
    """
    fname = "a.mmap.mma"
    nose.tools.assert_raises(ValueError, mmapping.load_memmap, fname)


def test_load_successful_2d():
    """
    Test if loading memmap for 2D array is successful.
    """
    fname = TWO_D_FNAME
    setup_2d_mmap()
    Yr, (d1, d2), T = mmapping.load_memmap(str(fname))
    teardown_2d_mmap()
    nose.tools.assert_equal((d1, d2), (10, 11))
    nose.tools.assert_equal(T, 12)
    nose.tools.assert_is_instance(Yr, np.memmap)


def test_load_successful_3d():
    """
    Test if loading memmap for 3D array is successful.
    """
    fname = THREE_D_FNAME
    setup_3d_mmap()
    Yr, (d1, d2, d3), T = mmapping.load_memmap(str(fname))
    teardown_3d_mmap()
    nose.tools.assert_equal((d1, d2, d3), (10, 11, 13))
    nose.tools.assert_equal(T, 12)
    nose.tools.assert_is_instance(Yr, np.memmap)


def setup_2d_mmap():
    """
    Create a 2D memmap for testing.
    """
    np.memmap(TWO_D_FNAME, mode="w+", dtype=np.float32, shape=(12, 10, 11, 13), order="F")


def teardown_2d_mmap():
    """
    Remove the 2D memmap after testing.
    """
    TWO_D_FNAME.unlink()


def setup_3d_mmap():
    """
    Create a 3D memmap for testing.
    """
    np.memmap(THREE_D_FNAME, mode="w+", dtype=np.float32, shape=(12, 10, 11, 13), order="F")


def teardown_3d_mmap():
    """
    Remove the 3D memmap after testing.
    """
    THREE_D_FNAME.unlink()