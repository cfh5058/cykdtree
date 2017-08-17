import numpy as np
import os
import time
from nose.tools import assert_raises, assert_equal
try:
    from mpi4py import MPI
except ImportError:  # pragma: w/o MPI
    MPI = None
import cykdtree
from cykdtree import PROF_ENABLED
from cykdtree.tests import make_points, make_points_neighbors
from cykdtree.tests.test_parallel_utils import MPITest
Nproc = (3, 4, 5)


def test_spawn_parallel(nproc=3, npts=20, ndim=2, periodic=False,
                        leafsize=3):  # pragma: w/ MPI
    if MPI is None:  # pragma: w/o MPI
        return
    else:  # pragma: w/ MPI
        pts, le, re, ls = make_points(npts, ndim, leafsize=leafsize)
        cykdtree.spawn_parallel(pts, nproc, leafsize=leafsize,
                                left_edge=le, right_edge=re,
                                periodic=periodic, with_coverage=True,
                                profile=True)
        profile = 'temp_prof.txt'
        cykdtree.spawn_parallel(pts, nproc, leafsize=leafsize,
                                left_edge=le, right_edge=re,
                                periodic=periodic, with_coverage=True,
                                profile=profile)
        if PROF_ENABLED:
            assert(os.path.isfile(profile))
            os.remove(profile)


@MPITest(Nproc, periodic=(False, True), ndim=(2, 3))
def test_PyParallelKDTree(periodic=False, ndim=2):  # pragma: w/ MPI
    pts, le, re, ls = make_points(20, ndim, leafsize=3)
    cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                              periodic=periodic)


@MPITest(3)
def test_PyParallelKDTree_errors(ndim=2):  # pragma: w/ MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pts, le, re, ls = make_points(20, ndim, leafsize=3)
    assert_raises(ValueError, cykdtree.PyParallelKDTree, pts,
                  le, re, leafsize=1)
    err_rank = 1
    if rank == err_rank:
        pts = np.ones((5, ndim), 'float64')
        assert_raises(AssertionError, cykdtree.PyParallelKDTree, pts,
                      le, re, leafsize=ls)
    else:
        assert_raises(Exception, cykdtree.PyParallelKDTree, pts,
                      le, re, leafsize=ls)


@MPITest(3)
def test_PyParallelKDTree_defaults(ndim=2):  # pragma: w/ MPI
    pts, le, re, ls = make_points(20, ndim, leafsize=3)
    cykdtree.PyParallelKDTree(pts, leafsize=ls)
    cykdtree.PyParallelKDTree(pts, nleaves=4)
    cykdtree.PyParallelKDTree(pts, leafsize=ls, periodic=np.ones(ndim, 'bool'))
    cykdtree.PyParallelKDTree(pts, leafsize=ls, use_sliding_midpoint=True)


@MPITest(3)
def test_PyParallelKDTree_properties():  # pragma: w/ MPI
    pts, le, re, ls = make_points(100, 2)
    T = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls)
    prop_list = ['local_npts', 'inter_npts', 'idx', 'inter_idx',
                 'left_edge', 'right_edge', 'domain_width',
                 'periodic_left', 'periodic_right']
    for p in prop_list:
        print(p)
        getattr(T, p)


@MPITest(Nproc, periodic=(False, True), ndim=(2, 3))
def test_consolidate(periodic=False, ndim=2):  # pragma: w/ MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pts, le, re, ls = make_points(20, ndim, leafsize=3)
    Tpara0 = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                       periodic=periodic)
    Tpara = Tpara0.consolidate()
    if rank == 0:
        if False:
            from cykdtree.plot import plot2D_serial
            plot2D_serial(Tpara, label_boxes=True,
                          plotfile='test_consolidate.png')
        Tseri = cykdtree.PyKDTree(pts, le, re, leafsize=ls,
                                  periodic=periodic)
        Tpara.assert_equal(Tseri, strict_idx=False)
    else:
        assert(Tpara is None)


@MPITest(Nproc, periodic=(False, True), ndim=(2, 3))
def test_search(periodic=False, ndim=2):  # pragma: w/ MPI
    pts, le, re, ls = make_points(100, ndim)
    tree = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                     periodic=periodic)
    if periodic:
        vals = [0, 0.5, 1.0]
    else:
        vals = [0, 0.5, 0.9]
    for v in vals:
        pos = v * np.ones(ndim, 'double')
        out = tree.get(pos)
        if out is not None:
            out.neighbors


@MPITest(3)
def test_search_errors(periodic=False, ndim=2):  # pragma: w/ MPI
    pts, le, re, ls = make_points(100, ndim)
    tree = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                     periodic=periodic)
    if not periodic:
        assert_raises(ValueError, tree.get, np.ones(ndim, 'double'))
    assert_raises(AssertionError, tree.get, np.zeros(ndim + 1, 'double'))


@MPITest(Nproc, periodic=(False, True))
def test_neighbors(periodic=False):  # pragma: w/ MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pts, le, re, ls, left_neighbors, right_neighbors = make_points_neighbors(
        periodic=periodic)
    tree = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                     periodic=periodic)
    if False:
        from cykdtree.plot import plot2D_parallel
        plotfile = 'test_neighbors.png'
        plot2D_parallel(tree, label_boxes=True, label_procs=True,
                        plotfile=plotfile)
    # Check left/right neighbors
    for leaf in tree.leaves.values():
        out_str = str(leaf.id)
        try:
            for d in range(tree.ndim):
                out_str += '\nleft:  {} {} {}'.format(
                    d, leaf.left_neighbors[d], left_neighbors[d][leaf.id])
                out_str += ' {}'.format(leaf.periodic_left)
                assert(len(left_neighbors[d][leaf.id]) ==
                       len(leaf.left_neighbors[d]))
                for i in range(len(leaf.left_neighbors[d])):
                    assert(left_neighbors[d][leaf.id][i] ==
                           leaf.left_neighbors[d][i])
                out_str += '\nright: {} {} {}'.format(
                    d, leaf.right_neighbors[d], right_neighbors[d][leaf.id])
                out_str += ' {}'.format(leaf.periodic_right)
                assert(len(right_neighbors[d][leaf.id]) ==
                       len(leaf.right_neighbors[d]))
                for i in range(len(leaf.right_neighbors[d])):
                    assert(right_neighbors[d][leaf.id][i] ==
                           leaf.right_neighbors[d][i])
        except:  # pragma: no cover
            time.sleep(rank)
            print(out_str)
            raise


@MPITest(3, periodic=(False, True))
def test_get_neighbor_ids(periodic=False, ndim=2):  # pragma: w/ MPI
    pts, le, re, ls = make_points(100, ndim)
    tree = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                     periodic=periodic)
    if periodic:
        vals = [0, 0.5, 1.0]
    else:
        vals = [0, 0.5, 0.9]
    for v in vals:
        pos = v * np.ones(ndim, 'float')
        tree.get_neighbor_ids(pos)


@MPITest(3)
def test_get_neighbor_ids_errors(periodic=False, ndim=2):  # pragma: w/ MPI
    pts, le, re, ls = make_points(100, ndim)
    tree = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                     periodic=periodic)
    if not periodic:
        assert_raises(ValueError, tree.get_neighbor_ids,
                      np.ones(ndim, 'double'))
    assert_raises(AssertionError, tree.get_neighbor_ids,
                  np.zeros(ndim + 1, 'double'))


@MPITest(Nproc, periodic=(False, True), ndim=(2, 3))
def test_consolidate_edges(periodic=False, ndim=2):  # pragma: w/ MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pts, le, re, ls = make_points(20, ndim, leafsize=3)
    Tpara = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                      periodic=periodic)
    LEpara, REpara = Tpara.consolidate_edges()
    if rank == 0:
        Tseri = cykdtree.PyKDTree(pts, le, re, leafsize=ls,
                                  periodic=periodic)
        LEseri, REseri = Tseri.consolidate_edges()
    else:
        LEseri, REseri = None, None
    LEseri, REseri = comm.bcast((LEseri, REseri), root=0)
    np.testing.assert_allclose(LEpara, LEseri)
    np.testing.assert_allclose(REpara, REseri)


@MPITest(Nproc, periodic=(False, True), ndim=(2, 3))
def test_consolidate_process_bounds(periodic=False, ndim=2):  # pragma: w/ MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    pts, le, re, ls = make_points(20, ndim, leafsize=3)
    Tpara = cykdtree.PyParallelKDTree(pts, le, re, leafsize=ls,
                                      periodic=periodic)
    LEpara, REpara = Tpara.consolidate_process_bounds()
    assert_equal(LEpara.shape, (size, ndim))
    assert_equal(REpara.shape, (size, ndim))
    np.testing.assert_allclose(LEpara[rank, :], Tpara.left_edge)
    np.testing.assert_allclose(REpara[rank, :], Tpara.right_edge)


def time_tree_construction(Ntime, LStime, ndim=2):  # pragma: w/ MPI
    if MPI is None:  # pragma: w/o MPI
        return
    else:  # pragma: w/ MPI
        pts, le, re, ls = make_points(Ntime, ndim, leafsize=LStime)
        t0 = time.time()
        cykdtree.PyParallelKDTree(pts, le, re, leafsize=LStime)
        t1 = time.time()
        print("{} {}D points, leafsize {}: took {} s".format(
            Ntime, ndim, LStime, t1 - t0))


def time_neighbor_search(Ntime, LStime, ndim=2):  # pragma: w/ MPI
    if MPI is None:  # pragma: w/o MPI
        return
    else:  # pragma: w/ MPI
        pts, le, re, ls = make_points(Ntime, ndim, leafsize=LStime)
        tree = cykdtree.PyParallelKDTree(pts, le, re, leafsize=LStime)
        t0 = time.time()
        tree.get_neighbor_ids(0.5 * np.ones(tree.ndim, 'double'))
        t1 = time.time()
        print("{} {}D points, leafsize {}: took {} s".format(
            Ntime, ndim, LStime, t1 - t0))


def test_time_tree_construction():
    time_tree_construction(100, 10)


def test_time_neighbor_search():
    time_neighbor_search(100, 10)
