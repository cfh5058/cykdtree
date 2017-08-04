from datetime import datetime
import cProfile
import pstats
import time
import signal
from subprocess import Popen, PIPE
from nose.tools import istest, nottest, assert_raises
try:
    from mpi4py import MPI
except ImportError:
    MPI=None
import numpy as np
import itertools
import sys
import os


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
            size_match = (xshape == yshape)
            assert((x <= y).all())
        else:
            assert(x <= y)
    except:
        if not size_match:
            raise AssertionError("Shape mismatch\n\n"+
                                 "x.shape: %s\ny.shape: %s\n" % 
                                 (str(x.shape), str(y.shape)))
        raise AssertionError("Variables are not less-equal ordered\n\n" +
                             "x: %s\ny: %s\n" % (str(x), str(y)))


def test_assert_less_equal():
    x = np.zeros(5)
    y = np.ones(5)
    assert_less_equal(x, y)
    assert_raises(AssertionError, assert_less_equal, y, x)
    assert_raises(AssertionError, assert_less_equal, x, np.ones(3))


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
        args_str += str(a)+","
    for k, v in kwargs.items():
        args_str += k+"="+str(v)+","
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
    func_cmds = function_call_lines(func, args, kwargs,
                                    with_coverage=with_coverage)
    cmd = ["mpirun", "-n", str(np), sys.executable, "-c",
           "'%s'" % (";".join(func_cmds))]
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


def test_call_subprocess():
    call_subprocess(1, assert_less_equal, [1, 5], {}, with_coverage=True)


def iter_dict(dicts):
    r"""Create a series of dicts by combining parameter sets.

    Args:
        dicts (iterable): Dictionaries that should be combined.

    Returns:
        tuple: Dictionaries with combined keyword values from the input.

    """
    try:
        return (dict(itertools.izip(dicts, x)) for x in
                itertools.product(*dicts.itervalues()))
    except AttributeError:
        # python 3
        return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


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
            for k,v in kwargs0.items():
                wrapped.__name__ += "_{}{}".format(k,v)
            return wrapped

        def func_param(*args, **kwargs):
            out = []
            for ipargs in iter_dict(pargs):
                out.append(pfunc(ipargs)(*args, **kwargs))
            return out

        func_param.__name__ = func.__name__

        return func_param

    return dec


def MPITest(Nproc, **pargs):
    r"""Decorator generator for tests that must be run with MPI.

    Args:
        Nproc (int, list, tuple): Number of processors or list/tuple of
            process counts that the test should be run with.
        \*\*pargs: Additional parameter values that the test should be
            parametrized by.

    Returns:
        func: Decorator function that calls the pass function with MPI.

    """
    if MPI is None:
        return lambda x: None

    if not isinstance(Nproc, (tuple, list)):
        Nproc = (Nproc,)
    max_size = max(Nproc)

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
                call_subprocess(s, func, args, kwargs, with_coverage=True)

            wrapped.__name__ = func.__name__
            return wrapped

        # Then just call the function
        else:
            @parametrize(**pargs)
            def try_func(*args, **kwargs):
                error_flag = np.array([0], 'int')
                try:
                    out = func(*args, **kwargs)
                except Exception as error:
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
    if MPI is not None:
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    else:
        size = 0
        rank = 0
    if (size == 0) or (rank == 0):
        np.random.set_state(rand_state)
        pts = np.random.rand(npts, ndim).astype('float64')
        left_edge = np.zeros(ndim, 'float64')
        right_edge = np.ones(ndim, 'float64')
    else:
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


def test_make_points_neighbors():
    make_points_neighbors()


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
    if MPI is not None:
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    else:
        size = 0
        rank = 0
    np.random.seed(seed)
    LE = 0.0
    RE = 1.0
    left_edge = LE*np.ones(ndim, 'float64')
    right_edge = RE*np.ones(ndim, 'float64')
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
                pts = np.random.normal(loc=(LE+RE)/2.0, scale=(RE-LE)/4.0,
                                       size=(npts, ndim))
                np.clip(pts, LE, RE)
            else:
                raise ValueError("Invalid 'distrib': {}".format(distrib))
    else:
        pts = None
        left_edge = None
        right_edge = None
    return pts, left_edge, right_edge, leafsize


@parametrize(npts=(-1,10), ndim=(2,3,4), distrib=('rand', 'uniform', 'normal'))
def test_make_points(npts=-1, ndim=2, distrib='rand'):
    make_points(npts, ndim, distrib=distrib)


def test_make_points_errors():
    assert_raises(ValueError, make_points, 10, 2, distrib='bad value')


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
    unique_str = datetime.today().strftime("%Y%j%H%M%S")
    pts, left_edge, right_edge, leafsize = make_points(npts, ndim,
                                                       leafsize=leafsize,
                                                       distrib=distrib)
    # Set keywords for multiprocessing version
    if nproc > 1:
        kwargs['suppress_final_output'] = suppress_final_output
        if profile:
            kwargs['profile'] = '{}_mpi_profile.dat'.format(unique_str)
    # Run
    if profile:
        pr = cProfile.Profile()
        t0 = time.time()
        pr.enable()
    out = make_tree(pts, nproc=nproc, left_edge=left_edge, right_edge=right_edge,
                    periodic=periodic, leafsize=leafsize, **kwargs)
    if profile:
        pr.disable()
        t1 = time.time()
        ps = pstats.Stats(pr)
        ps.add(kwargs['profile'])
        if os.path.isfile(kwargs['profile']):
            os.remove(kwargs['profile'])
        if isinstance(profile, str):
            ps.dump_stats(profile)
            print("Stats saved to {}".format(profile))
        else:
            sort_key = 'tottime'
            ps.sort_stats(sort_key).print_stats(25)
            # ps.sort_stats(sort_key).print_callers(5)
            print("{} s according to 'time'".format(t1-t0))
        return ps    


def test_run_test(npts=10, ndim=2, nproc=2, profile='temp_file.dat'):
    run_test(npts, ndim, nproc=nproc, profile=profile)
    assert(os.path.isfile(profile))
    os.remove(profile)


from cykdtree.tests import test_utils
from cykdtree.tests import test_kdtree
from cykdtree.tests import test_plot
from cykdtree.tests import test_parallel_kdtree
from cykdtree.tests import scaling
from cykdtree.tests import test_scaling

__all__ = ["MPITest", "test_utils", "test_kdtree",
           "test_parallel_kdtree", "test_plot", "make_points",
           "make_points_neighbors", "run_test", "scaling",
           "test_scaling"]
