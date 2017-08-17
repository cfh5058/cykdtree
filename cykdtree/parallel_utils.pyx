import numpy as np
cimport numpy as np
import sys
from subprocess import Popen, PIPE
cimport cython
from mpi4py import MPI
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libcpp cimport bool as cbool
from cpython cimport bool as pybool
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t


def function_call_lines(func, args, kwargs, with_coverage=False):
    r"""Get a list of lines required to run a function.

    Args:
        func (obj): Function object that should be run.
        args (list): List of function arguments.
        kwargs (dict): Dictionary of function keyword arguments.
        with_coverage (bool, optional): If True, lines will be added that
            enable coverage. Defaults to False.

    Returns:
        list: A list of strings containing the necessary lines to
            run the function.

    """
    cmds = []
    # Create string with arguments & kwargs
    args_str = ""
    for a in args:
        if isinstance(a, str):
            args_str += "\"%s\"" % a
        else:
            args_str += str(a)
        args_str += ","
    for k, v in kwargs.items():
        args_str += k+"="
        if isinstance(v, str):
            args_str += "\"%s\"" % v
        else:
            args_str += str(v)
        args_str += ","
    if args_str.endswith(","):
        args_str = args_str[:-1]
    # Coverage setup
    if with_coverage:
        cmds += ["from coverage import Coverage",
                 "cov = Coverage(auto_data=True)",
                 "cov.start()"]
    # Commands to run function
    cmds += ["from %s import %s" % (func.__module__, func.__name__),
             "%s(%s)" % (func.__name__, args_str)]
    # Coverage teardown
    if with_coverage:
        cmds += ["cov.stop()"]
    return cmds


def call_subprocess(np, func, args, kwargs, with_coverage=False):
    r"""Run a function call in parallel using mpirun.

    Args:
        np (int): Number of processes to run on.
        func (obj): Function object that should be run.
        args (list): List of function arguments.
        kwargs (dict): Dictionary of function keyword arguments.
        with_coverage (bool, optional): If True, coverage data for the
            executed code will be added to .coverage. Defaults to False.

    Returns:
        str: Output from the executed code.

    Raises:
        Exception: If there is an error on the spawned MPI process.

    """
    func_cmd = ";".join(function_call_lines(func, args, kwargs,
                                            with_coverage=with_coverage))
    cmd = ["mpirun", "-n", str(np), sys.executable, "-c",
           "'%s'" % func_cmd]
    cmd = ' '.join(cmd)
    print('Running the following command:\n%s' % cmd)
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    output, err = p.communicate()
    exit_code = p.returncode
    print(output.decode('utf-8'))
    if exit_code != 0:
        print(err.decode('utf-8'))
        raise Exception("Error on spawned process. See output.")
        # return None
    return output.decode('utf-8')


def py_parallel_distribute(np.ndarray[np.float64_t, ndim=2] pts0 = None):
    r"""Split points between all processes in the world.

    Args:
        pts0 (np.ndarray of np.float64): Array of points that should be split
            among the processes. This should only be passed to process 0.

    Returns:
        tuple(np.ndarray of float64, np.ndarray of uint64): The positions and
            original indices of those positions assigned to this processes.

    """
    cdef object comm = MPI.COMM_WORLD
    cdef int size = comm.Get_size()
    cdef int rank = comm.Get_rank()
    cdef uint64_t npts = 0
    cdef uint32_t ndim = 0
    cdef double *ptr_pts = NULL
    cdef uint64_t *ptr_idx = NULL
    cdef uint64_t[:] idx0
    if rank == 0:
        assert(pts0 is not None)
        npts = pts0.shape[0]
        ndim = pts0.shape[1]
        idx0 = np.arange(npts).astype('uint64')
        ptr_idx = &idx0[0]
        ptr_pts = &pts0[0,0]
    else:
        assert(pts0 is None)
    ndim = comm.bcast(ndim, root=0)
    cdef uint64_t nout;
    nout = parallel_distribute(&ptr_pts, &ptr_idx, npts, ndim)
    if nout > 0:
        assert(ptr_pts != NULL)
        assert(ptr_idx != NULL)
    # Memory view on pointers (memory may not be freed)
    # cdef np.float64_t[:,:] pts
    # cdef np.uint64_t[:] idx
    # pts = <np.float64_t[:nout, :ndim]> ptr_pts
    # idx = <np.uint64_t[:nout]> ptr_idx
    # Direct construction (ensures memory freed)
    cdef np.ndarray[np.float64_t, ndim=2] pts = np.empty((nout, ndim), 'float64')
    cdef uint64_t[:] idx = np.empty((nout,), 'uint64')
    cdef uint64_t i
    cdef uint32_t d
    for i in range(nout):
        idx[i] = ptr_idx[i]
        for d in range(ndim):
            pts[i,d] = ptr_pts[i*ndim+d]
    if ptr_pts != NULL:
        free(ptr_pts)
    if ptr_idx != NULL:
        free(ptr_idx)
    return (pts, np.asarray(idx))


def py_parallel_pivot_value(np.ndarray[np.float64_t, ndim=2] pts,
                            np.uint32_t pivot_dim):
    r"""Determine the pivot using median of medians across a pool of processes
    along a specified dimension.

    Args:
        pts (np.ndarray of float64): Positions on this process.
        pivot_dim (uint32): Dimension that median of medians should be performed
            along.

    Returns:
        float64: Median of medians across pool of processes.
    
    """
    cdef uint64_t npts = pts.shape[0]
    cdef uint32_t ndim = pts.shape[1]
    cdef int64_t l = 0
    cdef int64_t r = npts-1
    cdef int i
    # Get pivot
    cdef np.float64_t pivot
    cdef uint64_t[:] idx = np.arange(npts).astype('uint64')
    cdef double *ptr_pts = NULL
    cdef uint64_t *ptr_idx = NULL
    if npts != 0:
        ptr_pts = &pts[0,0]
        ptr_idx = &idx[0]
    pivot = parallel_pivot_value(ptr_pts, ptr_idx,
                                 ndim, pivot_dim, l, r);
    return pivot


def py_parallel_select(np.ndarray[np.float64_t, ndim=2] pts,
                       np.uint32_t pivot_dim, np.int64_t t):
    r"""Get the indices required to partition coordiantes such that the first
    q elements in pos[:,d] on each process cummulativly contain the smallest
    t elements in pos[:,d] across all processes. 

    Args:
        pts (np.ndarray of float64): Positions on this process.
        pivot_dim (uint32): Dimension that median of medians should be performed
            along.
        t (int64): Number of smallest elements in positions across all
            processes that should be partitioned.

    Returns:
        tuple(int64, float64, np.ndarray of uint64): Max index (q)  of points
            on this process that fall in the smallest t points overall, the
            value of element t (whether its on this process or not), and the
            index required to order the points to put the smallest ones first.

    """
    cdef uint64_t npts = pts.shape[0]
    cdef uint32_t ndim = pts.shape[1]
    cdef int64_t l = 0
    cdef int64_t r = npts-1
    cdef int i
    cdef double pivot_val = 0.0
    # Get pivot
    cdef uint64_t[:] idx = np.arange(npts).astype('uint64')
    cdef double *ptr_pts = NULL
    cdef uint64_t *ptr_idx = NULL
    if npts != 0:
        ptr_pts = &pts[0,0]
        ptr_idx = &idx[0]
    cdef int64_t q = parallel_select(ptr_pts, ptr_idx,
                                     ndim, pivot_dim, l, r, t, pivot_val);
    return q, pivot_val, idx


def py_parallel_split(np.ndarray[np.float64_t, ndim=2] pts, np.int64_t t,
                      np.ndarray[np.float64_t, ndim=1] mins = None,
                      np.ndarray[np.float64_t, ndim=1] maxs = None):
    r"""Get the indices required to partition coordinates such that the first
    q elements in pos on each process cummulativly contain the smallest
    t elements in pos along the largest dimension.

    Args:
        pts (np.ndarray of float64): Positions on this process.
        t (int64): Number of smallest elements in positions across all
            processes that should be partitioned.
        mins (np.ndarray of float64, optional): (m,) array of mins for this
            process. Defaults to None and is set to mins of pos along each
            dimension.
        maxs (np.ndarray of float64, optional): (m,) array of maxs for this
            process. Defaults to None and is set to maxs of pos along each
            dimension.

    Returns:
        tuple(int64, float64, np.ndarray of uint64): Max index (q)  of points
            on this process that fall in the smallest t points overall, the
            dimension the split was performed over, the value of element t
            (whether its on this process or not), and the index required to
            order the points to put the smallest ones first.

    """
    cdef uint64_t npts = pts.shape[0]
    cdef uint32_t ndim = pts.shape[1]
    cdef uint64_t Lidx = 0
    cdef int i
    # Get pivot
    cdef uint64_t[:] idx = np.arange(npts).astype('uint64')
    cdef double *ptr_pts = <double*>malloc(npts*ndim*sizeof(double))
    cdef uint64_t *ptr_idx = <uint64_t*>malloc(npts*sizeof(uint64_t))
    # cdef double *ptr_pts = NULL
    # cdef uint64_t *ptr_idx = NULL
    cdef double *ptr_mins = NULL
    cdef double *ptr_maxs = NULL
    if (npts != 0) and (ndim != 0):
        if mins is None:
            mins = np.min(pts, axis=0)
        if maxs is None:
            maxs = np.max(pts, axis=0)
        memcpy(ptr_pts, &pts[0,0], npts*ndim*sizeof(double))
        memcpy(ptr_idx, &idx[0], npts*sizeof(uint64_t))
        # ptr_pts = &pts[0,0]
        # ptr_idx = &idx[0]
        ptr_mins = &mins[0]
        ptr_maxs = &maxs[0]
    cdef int64_t q = 0
    cdef double split_val = 0.0
    cdef uint32_t dsplit = parallel_split(ptr_pts, ptr_idx, Lidx, npts, ndim,
                                          ptr_mins, ptr_maxs, q, split_val)
    # Array version
    cdef np.ndarray[np.uint64_t, ndim=1] new_idx
    cdef np.uint64_t j
    new_idx = np.empty((npts,), 'uint64')
    for j in range(npts):
        new_idx[j] = ptr_idx[j]
    free(ptr_idx)
    free(ptr_pts)
    return q, dsplit, split_val, new_idx


def py_redistribute_split(np.ndarray[np.float64_t, ndim=2] pts,
                          np.ndarray[np.uint64_t, ndim=1] idx,
                          np.ndarray[np.float64_t, ndim=1] mins = None,
                          np.ndarray[np.float64_t, ndim=1] maxs = None,
                          int split_rank = -1):
    r"""Repartition the points between processes such that the lower half
    of the processes have the lower half of the points as split along the
    largest dimension.

    Args:
        pts (np.ndarray of float64): Positions on this process.
        idx (np.ndarray of uint64): Original indices of positions on this
            process.
        mins (np.ndarray of float64, optional): (m,) array of mins for this
            process. Defaults to None and is set to mins of pos along each
            dimension.
        maxs (np.ndarray of float64, optional): (m,) array of maxs for this
            process. Defaults to None and is set to maxs of pos along each
            dimension.
        split_rank (int, optional): Rank that processes should be split at
            when repartitioning the points. Processes with ranks smaller
            than split_rank will contain points to the left of the split
            and processes with ranks greater than or equal to split_rank
            will contain points to the right of the split.

    Returns:
        tuple(np.ndarray of double, np.ndarray of uint64): The positions
            and original indices of the points on this process after the
            redistribution.

    """
    cdef object comm = MPI.COMM_WORLD
    cdef int size = comm.Get_size()
    cdef int rank = comm.Get_rank()
    if split_rank >= size:
        raise ValueError("split_rank must be smaller than the communicator " +
                         "size (%d)." % size)
    cdef uint64_t npts = pts.shape[0]
    cdef uint32_t ndim = pts.shape[1]
    cdef uint64_t Lidx = 0
    # Get pivot
    cdef double *ptr_pts = <double*>malloc(npts*ndim*sizeof(double))
    cdef uint64_t *ptr_idx = <uint64_t*>malloc(npts*sizeof(uint64_t))
    cdef double *ptr_mins = NULL
    cdef double *ptr_maxs = NULL
    if (npts != 0) and (ndim != 0):
        if mins is None:
            mins = np.min(pts, axis=0)
        if maxs is None:
            maxs = np.max(pts, axis=0)
        memcpy(ptr_pts, &pts[0,0], npts*ndim*sizeof(double))
        memcpy(ptr_idx, &idx[0], npts*sizeof(uint64_t))
        ptr_mins = &mins[0]
        ptr_maxs = &maxs[0]
    cdef int64_t split_idx = -1
    cdef uint32_t split_dim = 0
    cdef double split_val = 0.0
    cdef uint64_t new_npts = redistribute_split(&ptr_pts, &ptr_idx, npts, ndim,
                                                ptr_mins, ptr_maxs,
                                                split_idx, split_dim, split_val,
                                                split_rank)
    # Array version
    cdef np.ndarray[np.float64_t, ndim=2] new_pts
    cdef np.ndarray[np.uint64_t, ndim=1] new_idx
    cdef np.uint64_t i
    cdef np.uint32_t d
    new_pts = np.empty((new_npts, ndim), 'float64')
    new_idx = np.empty((new_npts,), 'uint64')
    for i in range(new_npts):
        new_idx[i] = ptr_idx[i]
        for d in range(ndim):
            new_pts[i, d] = ptr_pts[i*ndim+d]
    free(ptr_idx)
    free(ptr_pts)
    # Memory view version
    # cdef np.float64_t[:,:] new_pts
    # cdef np.uint64_t[:] new_idx
    # new_pts = <np.float64_t[:new_npts, :ndim]> ptr_pts
    # new_idx = <np.uint64_t[:new_npts]> ptr_idx
    return new_pts, new_idx, split_idx, split_dim, split_val


def py_calc_split_rank(int size, pybool split_left = None):
    r"""Determine the minimum rank in the right half of the processor split.

    Args:
        size (int): The size of the communicator that will be split.
        split_left (bool): If True, the middle process in an odd sized 
            communicator will be left of the split. If False, it will be
            right of the split. Defaults to None and uses the default at the
            c++ level.

    Returns:
        int: The rank of the first process in the right split.

    """
    cdef int split_rank
    cdef cbool c_split_left = <cbool>False
    if split_left is None:
        split_rank = calc_split_rank(size)
    else:
        c_split_left = <cbool>split_left
        split_rank = calc_split_rank(size, c_split_left)
    return split_rank

def py_calc_rounds():
    r"""Determine the number of rounds of splits required to populate all
    process in the world communicator and the round that this process would
    become the root of its subset.

    Returns:
        tuple(int, int): The number of rounds of splits required to populate
            all processes in the world communicator and the round that this
            process would become the root of a process subset.

    """
    cdef int nrounds, src_round
    src_round = 0
    nrounds = calc_rounds(src_round)
    return (nrounds, src_round)


def py_kdtree_parallel_distribute(np.ndarray[np.float64_t, ndim=2] pts = None):
    r"""Distribute a balanced number of points to each process using a kdtree
    structure.

    Args:
        pts (np.ndarray of np.float64): Array of points that should be split
            among the processes. This should only be passed to process 0.

    Returns:
        tuple(np.ndarray of float64, np.ndarray of uint64): Positions on this
            process and the indices of those positions in the input array
            on process 0.

    """
    cdef object comm = MPI.COMM_WORLD
    cdef int size = comm.Get_size()
    cdef int rank = comm.Get_rank()
    cdef uint64_t npts = 0
    cdef uint32_t ndim = 0
    cdef uint64_t Lidx = 0
    cdef double *ptr_le = <double*>malloc(ndim*sizeof(double));
    cdef double *ptr_re = <double*>malloc(ndim*sizeof(double));
    cdef cbool *ptr_ple = <cbool*>malloc(ndim*sizeof(cbool));
    cdef cbool *ptr_pre = <cbool*>malloc(ndim*sizeof(cbool));
    cdef double *ptr_min = <double*>malloc(ndim*sizeof(double));
    cdef double *ptr_max = <double*>malloc(ndim*sizeof(double));
    cdef uint32_t d;
    cdef exch_rec src;
    cdef vector[exch_rec] dst;
    if rank == 0:
        assert(pts is not None)
        npts = pts.shape[0]
        ndim = pts.shape[1]
        le = np.min(pts, axis=0)
        re = np.max(pts, axis=0)
        ple = np.zeros(ndim, 'bool')
        pre = np.zeros(ndim, 'bool')
        for d in range(ndim):
            ptr_le[d] = le[d]
            ptr_re[d] = re[d]
            ptr_ple[d] = <cbool>ple[d]
            ptr_pre[d] = <cbool>pre[d]
            ptr_min[d] = le[d]
            ptr_max[d] = re[d]
    else:
        assert(pts is None)
    ndim = comm.bcast(ndim, root=0)
    cdef uint64_t[:] idx = np.arange(npts).astype('uint64')
    # Set pointers
    cdef double *ptr_pts = NULL
    cdef uint64_t *ptr_idx = NULL
    if (npts != 0) and (ndim != 0):
        ptr_pts = &pts[0,0]
        ptr_idx = &idx[0]
    # Copy points (unnecessary if distribute allocates different mem block)
    # cdef double *ptr_pts
    # cdef uint64_t *ptr_idx
    # ptr_pts = <double*>malloc(npts*ndim*sizeof(double))
    # ptr_idx = <uint64_t*>malloc(npts*sizeof(uint64_t))
    # if (npts != 0) and (ndim != 0):
    #     memcpy(ptr_pts, &pts[0,0], npts*ndim*sizeof(double))
    #     memcpy(ptr_idx, &idx[0], npts*sizeof(uint64_t))
    # Distribute
    cdef uint64_t new_npts = kdtree_parallel_distribute(
        &ptr_pts, &ptr_idx, npts, ndim,
        ptr_le, ptr_re, ptr_ple, ptr_pre, ptr_min, ptr_max,
        src, dst)
    # Array version
    cdef np.ndarray[np.float64_t, ndim=2] new_pts
    cdef np.uint64_t[:] new_idx
    cdef np.ndarray[np.float64_t, ndim=1] new_le
    cdef np.ndarray[np.float64_t, ndim=1] new_re
    # cdef np.ndarray[bool, ndim=1] new_ple
    # cdef np.ndarray[bool, ndim=1] new_pre
    cdef np.uint64_t i
    new_pts = np.empty((new_npts, ndim), 'float64')
    new_idx = np.empty((new_npts,), 'uint64')
    new_le = np.empty(ndim, 'float64')
    new_re = np.empty(ndim, 'float64')
    new_ple = np.zeros(ndim, 'bool')
    new_pre = np.zeros(ndim, 'bool')
    for i in range(new_npts):
        new_idx[i] = ptr_idx[i]
        for d in range(ndim):
            new_pts[i, d] = ptr_pts[i*ndim+d]
    for d in range(ndim):
        new_le[d] = ptr_le[d]
        new_re[d] = ptr_re[d]
        new_ple[d] = <pybool>ptr_ple[d]
        new_pre[d] = <pybool>ptr_pre[d]
    free(ptr_idx)
    free(ptr_pts)
    free(ptr_le)
    free(ptr_re)
    free(ptr_ple)
    free(ptr_pre)
    free(ptr_min)
    free(ptr_max)
    # Memory view version
    # cdef np.float64_t[:,:] new_pts
    # cdef np.uint64_t[:] new_idx
    # new_pts = <np.float64_t[:new_npts, :ndim]> ptr_pts
    # new_idx = <np.uint64_t[:new_npts]> ptr_idx
    return new_pts, np.asarray(new_idx), new_le, new_re, new_ple, new_pre



