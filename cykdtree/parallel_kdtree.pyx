import cython
import numpy as np
cimport numpy as np
from datetime import datetime
import os
import traceback
import signal
import cProfile
import pstats
from subprocess import Popen, PIPE
from mpi4py import MPI
from cykdtree.backwards import dump_to_pickle, load_from_pickle
from libc.stdlib cimport malloc, free
from libcpp cimport bool as cbool
from cpython cimport bool as pybool
from libc.stdint cimport uint32_t, uint64_t, int32_t, int64_t
from cykdtree.parallel_utils import call_subprocess
from cykdtree import PROF_ENABLED


def spawn_parallel(#np.ndarray[np.float64_t, ndim=2] pts, int nproc,
                   pts, nproc, 
                   with_coverage=False, **kwargs):
    r"""Spawn processes to construct a tree in parallel and then
    return the consolidated tree to the calling process.

    Args:
        pts (np.ndarray of float64): (n,m) Array of n mD points.
        nproc (int): The number of MPI processes that should be spawned.
        with_coverage (bool, optional): If True, coverage data for the
            executed code will be added to .coverage. Defaults to False.
        profile (str, optional): If set to a file path, cProfile timing
            statistics are saved to it. If set to True, the 25 most
            costly calls are printed. Defaults to False.
        suppress_final_output (bool, optional): If True, the tree is not
            saved. Defaults to False.
        \*\*kwargs: Additional keyword arguments are passed to the appropriate
            class for constructuing the tree. 

    Returns:
        T (:class:`cykdtree.PyKDTree`): KDTree object.

    Raises:
        AssertionError: If the input file cannot be created.
        AssertionError: If the output file does not exist.

    """
    unique_str = datetime.today().strftime("%Y%j%H%M%S")
    finput = 'input_%s.dat' % unique_str
    foutput = 'output_%s.dat' % unique_str
    # Save input to a file
    dump_to_pickle(finput, [pts, kwargs])
    # Spawn in parallel
    out = call_subprocess(nproc, parallel_worker,
                          [finput, foutput], {},
                          with_coverage=with_coverage)
    # Read tree
    if not kwargs.get("suppress_final_output", False):
        assert(os.path.isfile(foutput))
        tree = PyKDTree.from_file(foutput)
    else:
        tree = None
    # Clean up
    os.remove(finput)
    if not kwargs.get("suppress_final_output", False):
        os.remove(foutput)
    # Return tree
    return tree
    

def parallel_worker(finput, foutput):
    r"""Load input on the root process, construct the tree in parallel,
    consolidate the tree on the root process, and save the tree to a file.

    Args:
        finput (str): Full path to location of the input file.
        foutput (str): Full path the file where the resulting tree should be
            saved.

    Keyword arguments from input file:
        profile (str, optional): If set to a file path, cProfile timing
            statistics are saved to it. If set to True, the 25 most
            costly calls are printed. Defaults to False.
        suppress_final_output (bool, optional): If True, the tree is not
            saved. Defaults to False.

    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # Read input
    if rank == 0:
        pts, kwargs = load_from_pickle(finput)
    else:
        pts, kwargs = (None, {})
    # profile = False
    # if PROF_ENABLED:
    #     profile = kwargs.pop("profile", False)
    profile = kwargs.pop("profile", False)
    if not PROF_ENABLED:
        profile = False
    suppress_final_output = kwargs.pop("suppress_final_output", False)
    suppress_final_output = comm.bcast(suppress_final_output, root=0)
    # Build & consolidate tree
    if profile:
        pr = cProfile.Profile()
        pr.enable()
    ptree = PyParallelKDTree(pts, **kwargs)
    # Consolidate
    if not suppress_final_output:
        tree = ptree.consolidate()
    if profile:
        pr.disable()
        if isinstance(profile, str):
            pr.dump_stats(profile)
        else:
            pstats.Stats(pr).sort_stats('time').print_stats(25)
    # Save output
    if not suppress_final_output and (rank == 0):
        tree.save(foutput)


cdef class PyParallelKDTree:
    r"""Object for constructing a KDTree in parallel. All arguments are only
    accepted and processed on rank 0 process.

    Args:
        pts (np.ndarray of double): (n,m) array of n coordinates in a
            m-dimensional domain.
        left_edge (np.ndarray of double, optional): (m,) domain minimum in each
            dimension. If not provided, it is determined from the points.
            Defaults to None.
        right_edge (np.ndarray of double, optional): (m,) domain maximum in
            each dimension. If not provided, it is determined from the points.
            Defaults to None.
        periodic (object, optional): Truth of the domain periodicity overall
            (if bool), or in each dimension (if np.ndarray). Defaults to
            `False`.            
        leafsize (int, optional): The maximum number of points that should be in
            a leaf. Defaults to 10000.  
        nleaves (int, optional): The number of leaves that should be in the
            resulting tree. If greater than 0, leafsize is adjusted to produce a
            tree with 2**(ceil(log2(nleaves))) leaves. The leafsize keyword
            argument is ignored if nleaves is greater zero. Defaults to 0. 
        use_sliding_midpoint (bool, optional): If True, the sliding midpoint
            rule is used to perform splits. Otherwise, the median is used.
            Defaults to False.

    Raises:
        ValueError: If `leafsize < 2`. This currectly segfaults. 
        AssertionError: If pts is not None on processes other than rank 0.

    Attributes:
        rank (int): MPI rank of the process.
        size (int): Size of the MPI world.
        npts (uint64): Number of points in the tree. Only set on rank 0.
        local_npts (uint64): Number of points on this process.
        inter_npts (uint64): Number of points on this process and all child
            processes.
        ndim (uint32): Number of dimensions points occupy.
        total_num_leaves (uint32): Total number of leaves in the tree on all
            processes.
        local_num_leaves (uint32): Number of leaves on this process.
        leafsize (uint32): Maximum number of points a leaf can have.
        leaves (dict of `cykdtree.PyNode`): Local tree leaves on this process.
        idx (np.ndarray of uint64): Indices sorting the local points on this
            process by leaf.
        inter_idx (np.ndarray of uint64): Indices sorting the points on this 
            process and all child processes by leaf.
        left_edge (np.ndarray of double): (m,) domain minimum in each dimension
            for the portion of the tree on this process.
        right_edge (np.ndarray of double): (m,) domain maximum in each dimension
            for the portion of the tree on this process.
        domain_width (np.ndarray of double): (m,) domain width in each dimension
            for the portion of the tree on this process.
        periodic_left (np.ndarray of bool): Truth of domain periodicity to the
            left in each dimension for the portion of the tree on this process. 
        periodic_right (np.ndarray of bool): Truth of domain periodicity to the
            right in each dimension for the portion of the tree on this process. 

    """

    def __cinit__(self):
        # Initialize everthing to NULL/0/None to prevent seg fault 
        self.size = 0
        self.rank = 0
        self._ptree = NULL
        self.npts = 0
        self.ndim = 0
        self.total_num_leaves = 0
        self.local_num_leaves = 0
        self.leafsize = 0
        self._left_edge = NULL
        self._right_edge = NULL
        self._periodic = NULL
        self.leaves = None
        self._idx = None

    def __init__(self, np.ndarray[double, ndim=2] pts = None,
                  np.ndarray[double, ndim=1] left_edge = None,
                  np.ndarray[double, ndim=1] right_edge = None,
                  object periodic = False, int leafsize = 10000,
                  int nleaves = 0, pybool use_sliding_midpoint = False):
        cdef object comm = MPI.COMM_WORLD
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        self.leafsize = leafsize
        cdef uint32_t i
        cdef object error = None
        cdef object error_flags = None
        cdef int error_flag = 0
        # Init & broadcast basic properties to all processes
        if self.rank == 0:
            self.ndim = pts.shape[1]
            self.npts = pts.shape[0]
            if nleaves > 0:
                nleaves = <int>(2**np.ceil(np.log2(<float>nleaves)))
                self.leafsize = self.npts/nleaves + 1
        self.ndim = comm.bcast(self.ndim, root=0)
        self.leafsize = comm.bcast(self.leafsize, root=0)
        if (self.leafsize < 2):
            # This is here to prevent segfault. The cpp code needs modified
            # to support leafsize = 1
            raise ValueError("Process %d: 'leafsize' cannot be smaller than 2." %
                             self.rank)
        # Determine bounds of domain
        try:
            if self.rank == 0:
                if left_edge is None:
                    left_edge = np.min(pts, axis=0)
                if right_edge is None:
                    right_edge = np.max(pts, axis=0)
                assert(left_edge.size == self.ndim)
                assert(right_edge.size == self.ndim)
                self._left_edge = <double *>malloc(self.ndim*sizeof(double))
                self._right_edge = <double *>malloc(self.ndim*sizeof(double))
                self._periodic = <cbool *>malloc(self.ndim*sizeof(cbool));
                for i in range(self.ndim):
                    self._left_edge[i] = left_edge[i]
                    self._right_edge[i] = right_edge[i]
                if isinstance(periodic, pybool):
                    for i in range(self.ndim):
                        self._periodic[i] = <cbool>periodic
                else:
                    for i in range(self.ndim):
                        self._periodic[i] = <cbool>periodic[i]
            else:
                assert(pts is None)
        except Exception as error:
            error_flag = 1
        # Handle errors
        error_flags = comm.allgather(error_flag)
        if sum(error_flags) > 0:
            if error_flag:
                raise error
                # traceback.print_exception(type(error), error, error.__traceback__)
            raise Exception("Process %d: There were errors on %d processes." % 
                            (self.rank, sum(error_flags)))
        # Create c object
        if self.rank == 0:
            self._make_tree(&pts[0,0], <cbool>use_sliding_midpoint)
        else:
            self._make_tree(NULL, <cbool>use_sliding_midpoint)
        # Create list of Python leaves 
        self.total_num_leaves = self._ptree.total_num_leaves
        self.local_num_leaves = self._ptree.tree.num_leaves
        self.leaves = {}
        cdef Node* leafnode
        cdef PyNode leafnode_py
        cdef object leaf_neighbors = None
        for k in xrange(self.local_num_leaves):
            leafnode = self._ptree.tree.leaves[k]
            leafnode_py = PyNode()
            leafnode_py._init_node(leafnode, self.local_num_leaves,
                                   self._ptree.total_domain_width)
            self.leaves[leafnode.leafid] = leafnode_py

    def __dealloc__(self):
        if self._left_edge != NULL:
            free(self._left_edge)
        if self._right_edge != NULL:
            free(self._right_edge)
        if self._periodic != NULL:
            free(self._periodic)
        if self._ptree != NULL:
            del self._ptree

    cdef void _make_tree(self, double *ptr_pts, bool use_sliding_midpoint):
        cdef uint64_t[:] idx = np.arange(self.npts).astype('uint64')
        cdef uint64_t *ptr_idx
        if self.npts > 0:
            ptr_idx = &idx[0]
        else:
            ptr_idx = NULL
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self._ptree = new ParallelKDTree(ptr_pts, ptr_idx,
                                             self.npts, self.ndim,
                                             self.leafsize, self._left_edge,
                                             self._right_edge, self._periodic,
                                             use_sliding_midpoint)
        self._idx = idx  # Ensure that memory view freed

    @property
    def local_npts(self):
        cdef uint64_t out = self._ptree.local_npts
        return out
    @property
    def inter_npts(self):
        cdef uint64_t out = self._ptree.inter_npts
        return out
    # @property
    # def pts(self):
    #     cdef np.float64_t[:,:] view
    #     view = <np.float64_t[:self.local_npts,:self.ndim]> self._ptree.all_pts
    #     return np.asarray(view)
    @property
    def idx(self):
        cdef np.uint64_t[:] view
        view = <np.uint64_t[:self.local_npts]> self._ptree.all_idx
        return np.asarray(view)
    @property
    def inter_idx(self):
        cdef np.uint64_t[:] view
        view = <np.uint64_t[:self.inter_npts]> self._ptree.all_idx
        return np.asarray(view)
    @property
    def left_edge(self):
        cdef np.float64_t[:] view
        view = <np.float64_t[:self.ndim]> self._ptree.local_domain_left_edge
        return np.asarray(view)
    @property
    def right_edge(self):
        cdef np.float64_t[:] view
        view = <np.float64_t[:self.ndim]> self._ptree.local_domain_right_edge
        return np.asarray(view)
    @property
    def domain_width(self):
        return self.right_edge - self.left_edge
    @property
    def periodic_left(self):
        cdef cbool[:] view
        view = <cbool[:self.ndim]> self._ptree.local_periodic_left
        return np.asarray(view)
    @property
    def periodic_right(self):
        cdef cbool[:] view
        view = <cbool[:self.ndim]> self._ptree.local_periodic_right
        return np.asarray(view)

    cdef object _get_neighbor_ids(self, np.ndarray[double, ndim=1] pos):
        cdef object comm = MPI.COMM_WORLD
        cdef object out = None
        assert(<uint32_t>len(pos) == self.ndim)
        cdef np.uint32_t i
        cdef vector[uint32_t] vout = self._ptree.get_neighbor_ids(&pos[0]);
        cdef pybool found = (vout.size() != 0)
        cdef object all_found = comm.allgather(found)
        if sum(all_found) == 0:
            raise ValueError("Position is not within the kdtree root node.")
        # elif sum(all_found) > 1:
        #     raise ValueError("Position is on more than one process.")
        if found:
            out = np.empty(vout.size(), 'uint32')
            for i in xrange(vout.size()):
                out[i] = vout[i]
        return out

    def get_neighbor_ids(self, np.ndarray[double, ndim=1] pos):
        r"""Return the IDs of leaves containing & neighboring a given position.
        If the position is not owned by this process, None is returned.

        Args:
            pos (np.ndarray of double): Coordinates.

        Returns:
            np.ndarray of uint32: Leaves containing/neighboring `pos`.

        Raises:
            ValueError: If pos is not contained withing the KDTree.

        """
        return self._get_neighbor_ids(pos)

    cdef object _get(self, np.ndarray[double, ndim=1] pos):
        cdef object comm = MPI.COMM_WORLD
        cdef object out = None
        assert(<uint32_t>len(pos) == self.ndim)
        cdef Node* leafnode = self._ptree.search(&pos[0])
        # cdef PyNode out = PyNode()
        cdef pybool found = (leafnode != NULL)
        cdef object all_found = comm.allgather(found)
        if sum(all_found) == 0:
            raise ValueError("Position is not within the kdtree root node.")
        # elif sum(all_found) > 1:
        #     raise ValueError("Position is on more than one process.")
        if found:
            out = self.leaves[leafnode.leafid]
        return out

    def get(self, np.ndarray[double, ndim=1] pos):
        r"""Return the leaf containing a given position. If the position is
        not owned by this process, None is returned.

        Args:
            pos (np.ndarray of double): Coordinates.

        Returns:
            :class:`cykdtree.PyNode`: Leaf containing `pos`.

        Raises:
            ValueError: If pos is not contained withing the KDTree.

        """
        return self._get(pos)


    cdef object _consolidate(self):
        cdef KDTree *stree = NULL
        cdef PyKDTree out = PyKDTree()
        # cdef ParallelKDTree *ptree = self._ptree
        stree = self._ptree.consolidate_tree()
        if self.rank == 0:
            out._init_tree(stree)
            return out
        else:
            assert(stree == NULL)
            return None

    def consolidate(self):
        r"""Return the serial KDTree on process 0.

        Returns:
            :class:`cykdtree.PyKDTree`: KDTree.

        """
        return self._consolidate()

    def consolidate_edges(self):
        r"""Return arrays of the left and right edges for all leaves in the
        tree on each process.

        Returns:
            tuple(np.ndarray of double, np.ndarray of double): The left (first
                array) and right (second array) edges of each leaf (1st array
                dimension), in each dimension (2nd array dimension).

        """
        cdef np.ndarray[np.float64_t, ndim=2] leaves_le
        cdef np.ndarray[np.float64_t, ndim=2] leaves_re
        leaves_le = np.empty((self.total_num_leaves, self.ndim), 'float64')
        leaves_re = np.empty((self.total_num_leaves, self.ndim), 'float64')
        self._ptree.consolidate_edges(&leaves_le[0,0], &leaves_re[0,0])
        return (leaves_le, leaves_re)
        
    def consolidate_process_bounds(self):
        r"""Returns arrays of the left and right edges for the section of the
        domain contained by each process.

        Returns:
            tuple(np.ndarray of double, np.ndarray of double): The left (first
                array) and right (second array) edges of each process (1st
                array dimension), in each dimension (2nd array dimension).

        """
        cdef np.ndarray[np.float64_t, ndim=2] all_lbounds
        cdef np.ndarray[np.float64_t, ndim=2] all_rbounds
        all_lbounds = np.empty((self.size, self.ndim), 'float64')
        all_rbounds = np.empty((self.size, self.ndim), 'float64')
        self._ptree.consolidate_process_bounds(&all_lbounds[0,0],
                                               &all_rbounds[0,0])
        return (all_lbounds, all_rbounds)
