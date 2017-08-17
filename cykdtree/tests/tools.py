from datetime import datetime
import cProfile
import pstats
import time
from nose.tools import nottest
try:
    from mpi4py import MPI
except ImportError:  # pragma: w/o MPI
    MPI = None
import numpy as np
import itertools
import os
from cykdtree import PROF_ENABLED


@nottest
def assert_less_equal(x, y):
    r"""Assert that x is less than or equal to y. Either variable can be a
    scalar or numpy array. If they are both arrays, they must have the same
    shape and elements are commpared on an element by element basis.

    Args:
        x (int, float, np.ndarray): Array on left side of <=.
        y (int, float, np.ndarray): Array on right side of <=.

    Raises:
        AssertionError: If x and y are both arrays but do not have the same
            shape.
        AssertionError: If x is not less than or equal to y.

    """
    size_match = True
    try:
        xshape = (1,)
        yshape = (1,)
        if (isinstance(x, np.ndarray) or isinstance(y, np.ndarray)):
            if isinstance(x, np.ndarray):
                xshape = x.shape
            if isinstance(y, np.ndarray):
                yshape = y.shape
        if xshape != (1,) and yshape != (1,):
            size_match = (xshape == yshape)
            assert(size_match)
        if xshape != (1,) or yshape != (1,):
            assert((x <= y).all())
        else:
            assert(x <= y)
    except AssertionError:
        if not size_match:
            raise AssertionError("Shape mismatch\n\n" +
                                 "x.shape: %s\ny.shape: %s\n" %
                                 (str(xshape), str(yshape)))
        raise AssertionError("Variables are not less-equal ordered\n\n" +
                             "x: %s\ny: %s\n" % (str(x), str(y)))


def iter_dict(dicts):
    r"""Create a series of dicts by combining parameter sets.

    Args:
        dicts (iterable): Dictionaries that should be combined.

    Returns:
        tuple: Dictionaries with combined keyword values from the input.

    """
    kord = sorted(dicts.keys())
    return (dict(zip(kord, x)) for x in itertools.product(
        *[dicts[k] for k in kord]))


def parametrize(**pargs):
    r"""Decorator for iterating over tests for combinations of parameters.

    Args:
        \*\*pargs (dict): Dictionary of values for parameters that should be
            iterated over.

    """
    for k in pargs.keys():
        if not isinstance(pargs[k], (tuple, list)):
            pargs[k] = (pargs[k],)

    def dec(func):

        def pfunc(kwargs0):
            # Wrapper so that name encodes parameters
            def wrapped(*args, **kwargs):
                kwargs.update(**kwargs0)
                return func(*args, **kwargs)
            wrapped.__name__ = func.__name__
            for k, v in kwargs0.items():
                wrapped.__name__ += "_{}{}".format(k, v)
            return wrapped

        def func_param(*args, **kwargs):
            out = []
            for ipargs in iter_dict(pargs):
                out.append(pfunc(ipargs)(*args, **kwargs))
            return out

        func_param.__name__ = func.__name__

        return func_param

    return dec


np.random.seed(100)
pts2 = np.random.rand(100, 2).astype('float64')
pts3 = np.random.rand(100, 3).astype('float64')
rand_state = np.random.get_state()
left_neighbors_x = [[],  # None
                    [0],
                    [1],
                    [2],
                    [],  # None
                    [],  # None
                    [4, 5],
                    [5]]
left_neighbors_y = [[],  # None
                    [],  # None
                    [],  # None
                    [],  # None
                    [0, 1],
                    [4],
                    [1, 2, 3],
                    [6]]
left_neighbors_x_periodic = [[3],
                             [0],
                             [1],
                             [2],
                             [6],
                             [6, 7],
                             [4, 5],
                             [5]]
left_neighbors_y_periodic = [[5],
                             [5, 7],
                             [7],
                             [7],
                             [0, 1],
                             [4],
                             [1, 2, 3],
                             [6]]


@nottest
def make_points_neighbors(periodic=False):
    r"""Return test points and accompanying neighbor solution in 2D.

    Args:
        periodic (bool, optional): If True, the neighbor solution assumes the
            domain is periodic. Defaults to False.

    Returns:
        pts (np.ndarray): Test points.
        left_edge (np.ndarray): Minimum bounds of the domain.
        right_edge (np.ndarray): Maximum bounds of the domain.
        leafsize (int): Size of leaves in the test tree.
        ln (list): List of neighbors to the left in the x and y dimensions.
        rn (list): List of neighbors to the right in the x and y dimensions.

    """
    ndim = 2
    npts = 50
    leafsize = 10
    if MPI is not None:  # pragma: w/ MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    else:  # pragma: w/o MPI
        size = 0
        rank = 0
    if (size == 0) or (rank == 0):
        np.random.set_state(rand_state)
        pts = np.random.rand(npts, ndim).astype('float64')
        left_edge = np.zeros(ndim, 'float64')
        right_edge = np.ones(ndim, 'float64')
    else:  # pragma: w/ MPI
        pts = None
        left_edge = None
        right_edge = None
    if periodic:
        lx = left_neighbors_x_periodic
        ly = left_neighbors_y_periodic
    else:
        lx = left_neighbors_x
        ly = left_neighbors_y
    num_leaves = len(lx)
    ln = [lx, ly]
    rn = [[[] for i in range(num_leaves)] for _
          in range(ndim)]
    for d in range(ndim):
        for i in range(num_leaves):
            for j in ln[d][i]:
                rn[d][j].append(i)
        for i in range(num_leaves):
            rn[d][i] = list(set(rn[d][i]))
    return pts, left_edge, right_edge, leafsize, ln, rn


@nottest
def make_points(npts, ndim, leafsize=10, distrib='rand', seed=100):
    r"""Create test points.

    Args:
        npts (int): Number of points that should be generated. If <=0, there
            will be 100 points with a known solution in 2 and 3 dimensions.
        ndim (int): Number of dimensions that points should have.
        leafsize (int, optional): Size of leaves that should be used. Defaults
            to 10.
        distrib (str, optional): The distribution that should be used to
            generate the points. Supported values include:
              'rand': [DEFAULT] Random distirbution.
              'uniform': Uniform distribution.
              'normal' or 'gaussian': Normal distribution.
        seed (int, optional): Seed to use for random number generation.
            Defaults to 100.

    Returns:
        pts (np.ndarray): Test points.
        left_edge (np.ndarray): Minimum bounds of the domain.
        right_edge (np.ndarray): Maximum bounds of the domain.
        leafsize (int): Size of leaves in the test tree. This will be equal to
            the input parameter unless npts <= 0 and then it will be set to the
            value for the known solution.

    Raises:
        ValueError: If the distrib is not one of the above supported values.

    """
    ndim = int(ndim)
    npts = int(npts)
    leafsize = int(leafsize)
    if MPI is not None:  # pragma: w/ MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    else:  # pragma: w/o MPI
        size = 0
        rank = 0
    np.random.seed(seed)
    LE = 0.0
    RE = 1.0
    left_edge = LE * np.ones(ndim, 'float64')
    right_edge = RE * np.ones(ndim, 'float64')
    if (size == 0) or (rank == 0):
        if npts <= 0:
            npts = 100
            leafsize = 10
            if ndim == 2:
                pts = pts2
            elif ndim == 3:
                pts = pts3
            else:
                pts = np.random.rand(npts, ndim).astype('float64')
        else:
            if distrib == 'rand':
                pts = np.random.rand(npts, ndim).astype('float64')
            elif distrib == 'uniform':
                pts = np.random.uniform(low=LE, high=RE, size=(npts, ndim))
            elif distrib in ('gaussian', 'normal'):
                pts = np.random.normal(loc=(LE + RE) / 2.0,
                                       scale=(RE - LE) / 4.0,
                                       size=(npts, ndim))
                np.clip(pts, LE, RE)
            else:
                raise ValueError("Invalid 'distrib': {}".format(distrib))
    else:  # pragma: w/ MPI
        pts = None
        left_edge = None
        right_edge = None
    return pts, left_edge, right_edge, leafsize


@nottest
def run_test(npts, ndim, nproc=0, distrib='rand', periodic=False, leafsize=10,
             profile=False, suppress_final_output=False, **kwargs):
    r"""Run a rountine with a designated number of points & dimensions on a
    selected number of processors.

    Args:
        npart (int): Number of particles.
        nproc (int): Number of processors.
        ndim (int): Number of dimensions.
        distrib (str, optional): Distribution that should be used when
            generating points. Defaults to 'rand'.
        periodic (bool, optional): If True, the domain is assumed to be
            periodic. Defaults to False.
        leafsize (int, optional): Maximum number of points that should be in
            an leaf. Defaults to 10.
        profile (bool, optional): If True cProfile is used. Defaults to False.
        suppress_final_output (bool, optional): If True, the final output
            from spawned MPI processes is suppressed. This is mainly for
            timing purposes. Defaults to False.

    """
    from cykdtree import make_tree
    if MPI is None and nproc > 1:  # pragma: w/o MPI
        raise RuntimeError("MPI could not be imported for parallel run.")
    unique_str = datetime.today().strftime("%Y%j%H%M%S")
    pts, left_edge, right_edge, leafsize = make_points(npts, ndim,
                                                       leafsize=leafsize,
                                                       distrib=distrib)
    # Set keywords for multiprocessing version
    if nproc > 1:  # pragma: w/ MPI
        kwargs['suppress_final_output'] = suppress_final_output
        if profile and PROF_ENABLED:
            kwargs['profile'] = '%s_mpi_profile.dat' % unique_str
    # Run
    if profile:
        pr = cProfile.Profile()
        t0 = time.time()
        pr.enable()
    make_tree(pts, nproc=nproc,
              left_edge=left_edge, right_edge=right_edge,
              periodic=periodic, leafsize=leafsize, **kwargs)
    if profile:
        pr.disable()
        t1 = time.time()
        ps = pstats.Stats(pr)
        if nproc > 1:  # pragma: w/ MPI
            para_prof = kwargs.get('profile', None)
            if isinstance(para_prof, str) and os.path.isfile(para_prof):
                ps.add(para_prof)
                os.remove(para_prof)
        if isinstance(profile, str):
            ps.dump_stats(profile)
            print("Stats saved to {}".format(profile))
        else:
            sort_key = 'tottime'
            ps.sort_stats(sort_key).print_stats(25)
            print("{} s according to 'time'".format(t1 - t0))
        return ps
