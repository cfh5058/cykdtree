import sys
import os
PY_MAJOR_VERSION = sys.version_info[0]
from cykdtree.kdtree import PyKDTree, PyNode
try:
    from cykdtree.parallel_kdtree import PyParallelKDTree, spawn_parallel, parallel_worker
    FLAG_MULTIPROC = True
except ImportError:  # pragma: w/o MPI
    PyParallelKDTree = spawn_parallel = parallel_worker = None
    FLAG_MULTIPROC = False
from cykdtree import tests, plot


def get_include():
    """
    Return the directory that contains the NumPy \\*.h header files.
    Extension modules that need to compile against NumPy should use this
    function to locate the appropriate include directory.
    Notes
    -----
    When using ``distutils``, for example in ``setup.py``.
    ::
        import numpy as np
        ...
        Extension('extension_name', ...
                include_dirs=[np.get_include()])
        ...
    """
    import cykdtree
    return os.path.dirname(cykdtree.__file__)


def make_tree(pts, nproc=0, **kwargs):
    r"""Build a KD-tree for a set of points.

    Args:
        pts (np.ndarray of float64): (n,m) Array of n mD points.
        nproc (int, optional): The number of MPI processes that should be
            spawned. If <2, no processes are spawned. Defaults to 0.
        \*\*kwargs: Additional keyword arguments are passed to the appropriate
            class for constructuing the tree.

    Returns:
        T (:class:`cykdtree.PyKDTree`): KDTree object.

    Raises:
        ValueError: If `pts` is not a 2D array.

    """
    # Check input
    if (pts.ndim != 2):
        raise ValueError("pts must be a 2D array of ND coordinates")
    # Parallel
    if nproc > 1 and FLAG_MULTIPROC:
        T = spawn_parallel(pts, nproc, **kwargs)
    # Serial
    else:
        T = PyKDTree(pts, **kwargs)
    return T


__all__ = ["PyKDTree", "PyNode", "tests", "get_include",
           "PyParallelKDTree", "plot", "make_tree"]
