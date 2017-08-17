from nose.tools import assert_raises
try:
    from mpi4py import MPI
except ImportError:  # pragma: w/o MPI
    MPI = None
import numpy as np
import os
from cykdtree.tests.tools import (assert_less_equal, make_points_neighbors,
                                  make_points, run_test, parametrize)
from cykdtree.tests.test_parallel_utils import MPITest


def test_assert_less_equal():
    x = np.zeros(5)
    y = np.ones(5)
    assert_less_equal(x, y)
    assert_raises(AssertionError, assert_less_equal, y, x)
    assert_raises(AssertionError, assert_less_equal, x, np.ones(3))


@MPITest(Nproc)
def test_make_points_neighbors():
    make_points_neighbors()


@MPITest(Nproc, npts=(-1, 10), ndim=(2, 3, 4),
         distrib=('rand', 'uniform', 'normal'))
def test_make_points(npts=-1, ndim=2, distrib='rand'):
    make_points(npts, ndim, distrib=distrib)


def test_make_points_errors():
    assert_raises(ValueError, make_points, 10, 2, distrib='bad value')


def test_run_test(npts=10, ndim=2, nproc=2, profile='temp_file.dat'):
    if MPI is None:  # pragma: w/o MPI
        assert_raises(RuntimeError, run_test, npts, ndim, nproc=nproc)
        nproc = 1
    run_test(npts, ndim, nproc=nproc, profile=True)
    run_test(npts, ndim, nproc=nproc, profile=profile)
    assert(os.path.isfile(profile))
    os.remove(profile)
