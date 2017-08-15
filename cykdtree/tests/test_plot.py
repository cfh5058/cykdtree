import os
from nose.tools import assert_raises
from cykdtree.plot import plot2D_serial, plot2D_parallel
from cykdtree.kdtree import PyKDTree
try:
    from cykdtree.parallel_kdtree import PyParallelKDTree
except ImportError:  # pragma: w/o MPI
    PyParallelKDTree = None
from cykdtree.tests import make_points


def test_plot2D_serial():
    fname_test = "test_plot2D_serial.png"
    pts, le, re, ls = make_points(10, 2, leafsize=2)
    tree = PyKDTree(pts, le, re, leafsize=ls)
    axs = plot2D_serial(tree, pts, title="Serial Test", plotfile=fname_test,
                        label_boxes=True)
    plot2D_serial(tree, [pts, pts], axs=axs, plotfile=fname_test)
    os.remove(fname_test)
    del axs


def test_plot2D_parallel():
    if PyParallelKDTree is None:  # pragma: w/o MPI
        assert_raises(RuntimeError, plot2D_parallel, None, None)
    else:  # pragma: w/ MPI
        fname_test = "test_plot2D_parallel.png"
        pts, le, re, ls = make_points(100, 2)
        tree = PyParallelKDTree(pts, le, re, leafsize=ls)
        axs = plot2D_parallel(tree, pts, title="Parallel Test", plotfile=fname_test,
                              label_boxes=True, label_procs=True)
        os.remove(fname_test)
        # plot2D_parallel(tree, pts, axs=axs)
        del axs


