import numpy as np
import time
import os
import tempfile
from nose.tools import assert_raises
import cykdtree
from cykdtree.tests import parametrize, make_points, make_points_neighbors


def test_get_include():
    idir = cykdtree.get_include()
    test_file = os.path.join(idir, 'c_kdtree.hpp')
    assert(os.path.isfile(test_file))


def test_make_tree(npts=100, ndim=2):
    assert_raises(ValueError, cykdtree.make_tree, np.ones((3,3,3)))
    pts, le, re, ls = make_points(npts, ndim)
    cykdtree.make_tree(pts, leafsize=ls, nproc=1)
    cykdtree.make_tree(pts, leafsize=ls, nproc=2)


def test_PyNode():
    n0 = cykdtree.kdtree.PyNode()
    pts, le, re, ls = make_points(100, 2)
    T = cykdtree.PyKDTree(pts, le, re, leafsize=ls)
    n1 = T.leaves[0]
    n0.init_node(n1)


def test_PyNode_properties():
    pts, le, re, ls = make_points(100, 2)
    T = cykdtree.PyKDTree(pts, le, re, leafsize=ls)
    n = T.leaves[0]
    prop_list = ['periodic_left', 'periodic_right',
                 'left_edge', 'right_edge', 'domain_width',
                 'slice', 'neighbors']
    for p in prop_list:
        print(p)
        getattr(n, p)
        eval('n.%s' % p)

def test_PyNode_repr():
    pts, le, re, ls = make_points(100, 2)
    T = cykdtree.PyKDTree(pts, le, re, leafsize=ls)
    print(T.leaves[0].__repr__())


@parametrize(npts=10, ndim=(2, 3), periodic=(False, True))
def test_PyKDTree(npts=100, ndim=2, periodic=False):
    pts, le, re, ls = make_points(npts, ndim, leafsize=3)
    cykdtree.PyKDTree(pts, le, re, leafsize=ls, periodic=periodic)


def test_PyKDTree_errors():
    pts, le, re, ls = make_points(100, 2)
    assert_raises(ValueError, cykdtree.PyKDTree, pts, le, re,
                  leafsize=1)


def test_PyKDTree_defaults():
    cykdtree.PyKDTree()
    pts, le, re, ls = make_points(10, 2)
    cykdtree.PyKDTree(pts=pts, nleaves=2)
    cykdtree.PyKDTree(pts, leafsize=ls)
    cykdtree.PyKDTree(pts, leafsize=ls, periodic=np.ones(2, 'bool'))
    cykdtree.PyKDTree(pts, nleaves=4)
    cykdtree.PyKDTree(pts, leafsize=ls, use_sliding_midpoint=True)


def test_PyKDTree_properties():
    pts, le, re, ls = make_points(100, 2)
    T = cykdtree.PyKDTree(pts, le, re, leafsize=ls)
    prop_list = ['periodic', 'idx',
                 'left_edge', 'right_edge', 'domain_width']
    for p in prop_list:
        print(p)
        getattr(T, p)
        eval('T.%s' % p)
    

@parametrize(strict_idx=(False, True))
def test_assert_equal(strict_idx=False):
    pts, le, re, ls = make_points(100, 2)
    T = cykdtree.PyKDTree(pts, le, re, leafsize=ls)
    T.assert_equal(T, strict_idx)
    T.assert_equal(T, strict_idx=strict_idx)


@parametrize(npts=100, ndim=(2, 3), periodic=(False, True))
def test_search(npts=100, ndim=2, periodic=False):
    pts, le, re, ls = make_points(npts, ndim)
    tree = cykdtree.PyKDTree(pts, le, re, leafsize=ls, periodic=periodic)
    pos_list = [le, (le+re)/2.]
    if periodic:
        pos_list.append(re)
    for pos in pos_list:
        leaf = tree.get(pos)
        leaf.neighbors


@parametrize(npts=100, ndim=(2, 3))
def test_search_errors(npts=100, ndim=2):
    pts, le, re, ls = make_points(npts, ndim)
    tree = cykdtree.PyKDTree(pts, le, re, leafsize=ls)
    assert_raises(ValueError, tree.get, re)


@parametrize(periodic=(False, True))
def test_neighbors(periodic=False):
    pts, le, re, ls, left_neighbors, right_neighbors = make_points_neighbors(
        periodic=periodic)
    tree = cykdtree.PyKDTree(pts, le, re, leafsize=ls, periodic=periodic)
    for leaf in tree.leaves:
        out_str = str(leaf.id)
        try:
            for d in range(tree.ndim):
                out_str += '\nleft:  {} {} {}'.format(d, leaf.left_neighbors[d],
                                               left_neighbors[d][leaf.id])
                assert(len(left_neighbors[d][leaf.id]) ==
                       len(leaf.left_neighbors[d]))
                for i in range(len(leaf.left_neighbors[d])):
                    assert(left_neighbors[d][leaf.id][i] ==
                           leaf.left_neighbors[d][i])
                out_str += '\nright: {} {} {}'.format(d, leaf.right_neighbors[d],
                                                right_neighbors[d][leaf.id])
                assert(len(right_neighbors[d][leaf.id]) ==
                       len(leaf.right_neighbors[d]))
                for i in range(len(leaf.right_neighbors[d])):
                    assert(right_neighbors[d][leaf.id][i] ==
                           leaf.right_neighbors[d][i])
        except:  # pragma: no cover
            for leaf in tree.leaves:
                print(leaf.id, leaf.left_edge, leaf.right_edge)
            print(out_str)
            raise


def test_leaf_idx():
    pts, le, re, ls = make_points(10, 2)
    tree = cykdtree.PyKDTree(pts, leafsize=ls)
    tree.leaf_idx(0)


def test_get_neighbor_ids_3(npts=100):
    pts, le, re, ls = make_points(npts, 2)
    tree = cykdtree.PyKDTree(pts, le, re, leafsize=ls)
    assert_raises(ValueError, tree.get_neighbor_ids_3, le)
    pts, le, re, ls = make_points(npts, 3)
    tree = cykdtree.PyKDTree(pts, le, re, leafsize=ls)
    tree.get_neighbor_ids_3(le)


@parametrize(npts=100, ndim=(2,3), periodic=(False, True))
def test_get_neighbor_ids(npts=100, ndim=2, periodic=False):
    pts, le, re, ls = make_points(npts, ndim)
    tree = cykdtree.PyKDTree(pts, le, re, leafsize=ls, periodic=periodic)
    pos_list = [le, (le+re)/2.]
    if periodic:
        pos_list.append(re)
    for pos in pos_list:
        tree.get_neighbor_ids(pos)


def test_consolidate_edges():
    pts, le, re, ls = make_points(10, 2)
    tree = cykdtree.PyKDTree(pts, leafsize=ls)
    tree.consolidate_edges()


def time_tree_construction(Ntime, LStime, ndim=2):
    pts, le, re, ls = make_points(Ntime, ndim, leafsize=LStime)
    t0 = time.time()
    cykdtree.PyKDTree(pts, le, re, leafsize=LStime)
    t1 = time.time()
    print("{} {}D points, leafsize {}: took {} s".format(Ntime, ndim, LStime, t1-t0))


def test_time_tree_construction():
    time_tree_construction(10, 2)


def time_neighbor_search(Ntime, LStime, ndim=2):
    pts, le, re, ls = make_points(Ntime, ndim, leafsize=LStime)
    tree = cykdtree.PyKDTree(pts, le, re, leafsize=LStime)
    t0 = time.time()
    tree.get_neighbor_ids(0.5*np.ones(tree.ndim, 'double'))
    t1 = time.time()
    print("{} {}D points, leafsize {}: took {} s".format(Ntime, ndim, LStime, t1-t0))


def test_time_neighbor_search():
    time_neighbor_search(10, 2)


def test_save_load():
    for periodic in (True, False):
        for ndim in range(1, 5):
            pts, le, re, ls = make_points(100, ndim)
            tree = cykdtree.PyKDTree(pts, le, re, leafsize=ls,
                                     periodic=periodic, data_version=ndim+12)
            with tempfile.NamedTemporaryFile() as tf:
                tree.save(tf.name)
                restore_tree = cykdtree.PyKDTree.from_file(tf.name)
                tree.assert_equal(restore_tree)
