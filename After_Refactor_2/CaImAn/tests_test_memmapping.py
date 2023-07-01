def get_caiman_datadir():
    return pathlib.Path(caiman_datadir())


def create_memmap_file(fname, shape):
    np.memmap(fname, mode="w+", dtype=np.float32, shape=shape, order="F")


TWO_D_SHAPE = (12, 10, 11, 13)
THREE_D_SHAPE = (12, 10, 11, 13)
TWO_D_FNAME = get_caiman_datadir() / "testdata/memmap__d1_10_d2_11_d3_1_order_F_frames_12_.mmap"
THREE_D_FNAME = get_caiman_datadir() / "testdata/memmap__d1_10_d2_11_d3_13_order_F_frames_12_.mmap"


def test_load_raises_wrong_ext():
    fname = "a.mmapp"
    try:
        mmapping.load_memmap(fname)
    except ValueError:
        assert True
    else:
        assert False


def test_load_raises_multiple_ext():
    fname = "a.mmap.mma"
    try:
        mmapping.load_memmap(fname)
    except ValueError:
        assert True
    else:
        assert False


def setup_memmap_file(fname, shape):
    create_memmap_file(fname, shape)


def teardown_memmap_file(fname):
    fname.unlink()


def setup_2d_mmap():
    setup_memmap_file(TWO_D_FNAME, TWO_D_SHAPE)


def teardown_2d_mmap():
    teardown_memmap_file(TWO_D_FNAME)


def setup_3d_mmap():
    setup_memmap_file(THREE_D_FNAME, THREE_D_SHAPE)


def teardown_3d_mmap():
    teardown_memmap_file(THREE_D_FNAME)


@nose.with_setup(setup_2d_mmap, teardown_2d_mmap)
def test_load_successful_2d():
    fname = get_caiman_datadir() / "testdata" / TWO_D_FNAME
    Yr, (d1, d2), T = mmapping.load_memmap(str(fname))
    assert (d1, d2) == (10, 11)
    assert T == 12
    assert isinstance(Yr, np.memmap)


@nose.with_setup(setup_3d_mmap, teardown_3d_mmap)
def test_load_successful_3d():
    fname = get_caiman_datadir() / "testdata" / THREE_D_FNAME
    Yr, (d1, d2, d3), T = mmapping.load_memmap(str(fname))
    assert (d1, d2, d3) == (10, 11, 13)
    assert T == 12
    assert isinstance(Yr, np.memmap)

def create_memmap_file(fname, shape, dtype=np.float32, order="F"):
    np.memmap(fname, mode="w+", dtype=dtype, shape=shape, order=order)


def test_load_raises_wrong_ext():
    fname = "a.mmapp"
    try:
        mmapping.load_memmap(fname)
    except ValueError:
        assert True
    else:
        assert False


def test_load_raises_multiple_ext():
    fname = "a.mmap.mma"
    try:
        mmapping.load_memmap(fname)
    except ValueError:
        assert True
    else:
        assert False


def setup_memmap_file(fname, shape, dtype=np.float32, order="F"):
    create_memmap_file(fname, shape, dtype, order)


def teardown_memmap_file(fname):
    fname.unlink()


def setup_2d_mmap():
    setup_memmap_file(TWO_D_FNAME, TWO_D_SHAPE)


def teardown_2d_mmap():
    teardown_memmap_file(TWO_D_FNAME)


def setup_3d_mmap():
    setup_memmap_file(THREE_D_FNAME, THREE_D_SHAPE)


def teardown_3d_mmap():
    teardown_memmap_file(THREE_D_FNAME)


@nose.with_setup(setup_2d_mmap, teardown_2d_mmap)
def test_load_successful_2d():
    fname = get_caiman_datadir() / "testdata" / TWO_D_FNAME
    Yr, (d1, d2), T = mmapping.load_memmap(str(fname))
    assert (d1, d2) == (10, 11)
    assert T == 12
    assert isinstance(Yr, np.memmap)


@nose.with_setup(setup_3d_mmap, teardown_3d_mmap)
def test_load_successful_3d():
    fname = get_caiman_datadir() / "testdata" / THREE_D_FNAME
    Yr, (d1, d2, d3), T = mmapping.load_memmap(str(fname))
    assert (d1, d2, d3) == (10, 11, 13)
    assert T == 12
    assert isinstance(Yr, np.memmap)