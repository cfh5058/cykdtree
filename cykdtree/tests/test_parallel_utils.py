import numpy as np
from nose.tools import assert_raises, assert_equal
try:  # pragma: w/ MPI
    from mpi4py import MPI
    from cykdtree import parallel_utils
except ImportError:  # pragma: w/o MPI
    MPI = None
    parallel_utils = None
from cykdtree.tests import assert_less_equal, parametrize
from cykdtree import utils
Nproc = (3, 4, 5)
Nproc_single = 3


def assert_with_keywords(v1, v2=0, v3=""):
    assert_less_equal(v1, v2)


def test_call_subprocess():  # pragma: w/ MPI
    if MPI is None:  # pragma: w/o MPI
        assert_with_keywords(0, v2=1)
    else:  # pragma: w/ MPI
        parallel_utils.call_subprocess(1, assert_with_keywords,
                                       [1], dict(v2=5, v3="test"),
                                       with_coverage=True)
        assert_raises(Exception, parallel_utils.call_subprocess, 1,
                      assert_with_keywords, [1], dict(v2=0, v3="test"))
                      

def MPITest(Nproc, **pargs):  # pragma: w/ MPI
    r"""Decorator generator for tests that must be run with MPI.

    Args:
        Nproc (int, list, tuple): Number of processors or list/tuple of
            process counts that the test should be run with.
        \*\*pargs: Additional parameter values that the test should be
            parametrized by.

    Returns:
        func: Decorator function that calls the pass function with MPI.

    """
    if MPI is None:  # pragma: w/o MPI
        return lambda x: None

    if not isinstance(Nproc, (tuple, list)):
        Nproc = (Nproc,)

    def dec(func):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        # print(size, Nproc, size in Nproc)

        # First do setup
        if (size not in Nproc):
            @parametrize(Nproc=Nproc)
            def wrapped(*args, **kwargs):
                s = kwargs.pop('Nproc', 1)
                parallel_utils.call_subprocess(s, func, args, kwargs,
                                               with_coverage=True)

            wrapped.__name__ = func.__name__
            return wrapped

        # Then just call the function
        else:
            @parametrize(**pargs)
            def try_func(*args, **kwargs):
                error_flag = np.array([0], 'int')
                try:
                    out = func(*args, **kwargs)
                except Exception:
                    import traceback
                    print(traceback.format_exc())
                    error_flag[0] = 1
                flag_count = np.zeros(1, 'int')
                comm.Allreduce(error_flag, flag_count)
                if flag_count[0] > 0:
                    raise Exception("Process %d: There were errors on %d processes." %
                                    (rank, flag_count[0]))
                return out
            return try_func
    return dec


def test_MPITest():  # pragma: w/ MPI
    def func_raise():
        raise Exception
    f = MPITest(1)(func_raise)
    assert_raises(Exception, f)


@MPITest(Nproc, ndim=(2, 3))
def test_parallel_distribute(ndim=2):  # pragma: w/ MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    npts = int(50)
    if rank == 0:
        pts = np.random.rand(npts, ndim).astype('float64')
    else:
        pts = None
    total_pts = comm.bcast(pts, root=0)
    local_pts, local_idx = parallel_utils.py_parallel_distribute(pts)
    npts_local = npts // size
    if rank < (npts % size):
        npts_local += 1
    assert_equal(local_pts.shape, (npts_local, ndim))
    assert_equal(local_idx.shape, (npts_local, ))
    np.testing.assert_array_equal(total_pts[local_idx], local_pts)


@MPITest(Nproc, ndim=(2, 3), npts=(10, 11, 50, 51))
def test_parallel_pivot_value(ndim=2, npts=50):  # pragma: w/ MPI
    npts = int(npts)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        pts = np.random.rand(npts, ndim).astype('float64')
    else:
        pts = None
    total_pts = comm.bcast(pts, root=0)
    local_pts, local_idx = parallel_utils.py_parallel_distribute(pts)
    pivot_dim = ndim - 1

    piv = parallel_utils.py_parallel_pivot_value(local_pts, pivot_dim)

    nmax = (7 * npts // int(10) + 6)
    assert(np.sum(total_pts[:, pivot_dim] < piv) <= nmax)
    assert(np.sum(total_pts[:, pivot_dim] > piv) <= nmax)

    # Not equivalent because each processes does not have multiple of 5 points
    # if rank == 0:
    #     pp, idx = utils.py_pivot(total_pts, pivot_dim)
    #     np.testing.assert_approx_equal(piv, total_pts[idx[pp], pivot_dim])


@MPITest(Nproc, ndim=(2, 3), npts=(10, 11, 50, 51))
def test_parallel_select(ndim=2, npts=50):  # pragma: w/ MPI
    total_npts = int(npts)
    pivot_dim = ndim - 1
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        total_pts = np.random.rand(total_npts, ndim).astype('float64')
    else:
        total_pts = None
    pts, orig_idx = parallel_utils.py_parallel_distribute(total_pts)
    npts = pts.shape[0]

    p = total_npts // 2 + total_npts % 2
    q, piv, idx = parallel_utils.py_parallel_select(pts, pivot_dim, p)
    assert_equal(idx.size, npts)

    total_pts = comm.bcast(total_pts, root=0)
    if npts != 0:
        med = np.median(total_pts[:, pivot_dim])
        if (total_npts % 2):
            np.testing.assert_approx_equal(piv, med)
        else:
            np.testing.assert_array_less(piv, med)
        if q >= 0:
            assert_less_equal(pts[idx[:(q + 1)], pivot_dim], piv)
            np.testing.assert_array_less(piv, pts[idx[(q + 1):], pivot_dim])
    # TODO: fix this
    # else:
    #     assert_equal(q, -1)


@MPITest(Nproc, ndim=(2, 3), npts=(10, 11, 50, 51))
def test_parallel_split(ndim=2, npts=50):  # pragma: w/ MPI
    total_npts = int(npts)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        total_pts = np.random.rand(total_npts, ndim).astype('float64')
    else:
        total_pts = None
    pts, orig_idx = parallel_utils.py_parallel_distribute(total_pts)
    npts = pts.shape[0]

    p = total_npts // 2 + total_npts % 2
    q, pivot_dim, piv, idx = parallel_utils.py_parallel_split(pts, p)
    assert_equal(idx.size, npts)

    total_pts = comm.bcast(total_pts, root=0)
    if npts != 0:
        med = np.median(total_pts[:, pivot_dim])
        if (total_npts % 2):
            np.testing.assert_approx_equal(piv, med)
        else:
            np.testing.assert_array_less(piv, med)
        if q >= 0:
            assert_less_equal(pts[idx[:(q + 1)], pivot_dim], piv)
            np.testing.assert_array_less(piv, pts[idx[(q + 1):], pivot_dim])
    # TODO: Actually check this
    # else:
    #     assert_equal(q, -1)

    if rank == 0:
        sq, sd, sidx = utils.py_split(total_pts)
        assert_equal(pivot_dim, sd)
        assert_equal(piv, total_pts[sidx[sq], sd])


@MPITest(Nproc, ndim=(2, 3), npts=(10, 11, 50, 51), split_left=(None, False, True))
def test_redistribute_split(ndim=2, npts=50, split_left=None):  # pragma: w/ MPI
    total_npts = npts
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if split_left is None:
        split_rank = -1
    else:
        split_rank = size // int(2)
        if split_left:
            split_rank += size % 2
    if rank == 0:
        total_pts = np.random.rand(total_npts, ndim).astype('float64')
    else:
        total_pts = None
    pts, orig_idx = parallel_utils.py_parallel_distribute(total_pts)
    npts = pts.shape[0]
    total_pts = comm.bcast(total_pts, root=0)

    new_pts, new_idx, sidx, sdim, sval = parallel_utils.py_redistribute_split(
        pts, orig_idx, split_rank=split_rank)
    # Assume split_left is default for split_rank == -1
    if split_rank < 0:
        split_rank = size // int(2) + size % 2

    assert_equal(new_pts.shape[0], new_idx.size)
    assert_equal(new_pts.shape[1], ndim)

    np.testing.assert_array_equal(new_pts, total_pts[new_idx, :])

    if rank < split_rank:
        assert_less_equal(new_pts[:, sdim], sval)
    else:
        np.testing.assert_array_less(sval, new_pts[:, sdim])

    med = np.median(total_pts[:, sdim])
    if (total_npts % 2):
        np.testing.assert_approx_equal(sval, med)
    else:
        np.testing.assert_array_less(sval, med)


@MPITest(Nproc_single, ndim=(2, ), npts=(10, ))
def test_redistribute_split_errors(ndim=2, npts=50):  # pragma: w/ MPI
    total_npts = npts
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        total_pts = np.random.rand(total_npts, ndim).astype('float64')
    else:
        total_pts = None
    pts, orig_idx = parallel_utils.py_parallel_distribute(total_pts)
    assert_raises(ValueError, parallel_utils.py_redistribute_split,
                  pts, orig_idx, split_rank=size)
    parallel_utils.py_redistribute_split(pts, orig_idx,
                                         mins=np.min(pts, axis=0),
                                         maxs=np.max(pts, axis=0))


def test_calc_split_rank():  # pragma: w/ MPI
    if MPI is None:
        return  # pragma: w/o MPI

    # Default split (currently left)
    assert_equal(parallel_utils.py_calc_split_rank(4), 2)
    assert_equal(parallel_utils.py_calc_split_rank(5), 3)
    # Left split
    assert_equal(parallel_utils.py_calc_split_rank(4, split_left=True), 2)
    assert_equal(parallel_utils.py_calc_split_rank(5, split_left=True), 3)
    # Right split
    assert_equal(parallel_utils.py_calc_split_rank(4, split_left=False), 2)
    assert_equal(parallel_utils.py_calc_split_rank(5, split_left=False), 2)


@MPITest(Nproc)
def test_calc_rounds():  # pragma: w/ MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Get answers
    ans_nrounds = int(np.ceil(np.log2(size))) + 1
    ans_src_round = 0
    curr_rank = rank
    curr_size = size
    while curr_rank != 0:
        split_rank = parallel_utils.py_calc_split_rank(curr_size)
        if curr_rank < split_rank:
            curr_size = split_rank
            curr_rank = curr_rank
        else:
            curr_size = curr_size - split_rank
            curr_rank = curr_rank - split_rank
        ans_src_round += 1
    # Test
    nrounds, src_round = parallel_utils.py_calc_rounds()
    assert_equal(nrounds, ans_nrounds)
    assert_equal(src_round, ans_src_round)


@MPITest(Nproc, ndim=(2, 3), npts=(10, 11, 50, 51))
def test_kdtree_parallel_distribute(ndim=2, npts=50):  # pragma: w/ MPI
    total_npts = npts
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        total_pts = np.random.rand(total_npts, ndim).astype('float64')
    else:
        total_pts = None
    pts, idx, le, re, ple, pre = parallel_utils.py_kdtree_parallel_distribute(total_pts)
    total_pts = comm.bcast(total_pts, root=0)
    assert_equal(pts.shape[0], idx.size)
    np.testing.assert_array_equal(pts, total_pts[idx, :])
    for d in range(ndim):
        assert_less_equal(pts[:, d], re[d])
        assert_less_equal(le[d], pts[:, d])
