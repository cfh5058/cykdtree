from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.sdist import sdist as _sdist
from subprocess import Popen, PIPE
import copy
import numpy
import os
import sys

PY_MAJOR_VERSION = sys.version_info[0]

# Check for ReadTheDocs/coverage flags
RTDFLAG = bool(os.environ.get('READTHEDOCS', None) == 'True')
COVFLAG = bool(os.environ.get('CYKDTREE_COVERAGE', None) == 'True')
PRFFLAG = bool(os.environ.get('CYKDTREE_PROFILE', None) == 'True')

# Check for Cython
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    raise ImportError('Cython is a required dependency of cykdtree')

ext_options = dict(language="c++",
                   include_dirs=[numpy.get_include()],
                   libraries=[],
                   extra_link_args=[],
                   extra_compile_args=["-std=c++03"])
cyt_options = {}

# Needed for line_profiler/coverage - disabled for production code
if not RTDFLAG and (COVFLAG or PRFFLAG):
    try:
        from Cython.Compiler.Options import directive_defaults
    except ImportError:
        # Update to cython
        from Cython.Compiler.Options import get_directive_defaults
        directive_defaults = get_directive_defaults()
    directive_defaults['profile'] = True
    directive_defaults['linetrace'] = True
    directive_defaults['binding'] = True
    cyt_options['compiler_directives'] = {'linetrace': True,
                                          'profile': True,
                                          'binding': True}
    ext_options['define_macros'] = [('CYTHON_TRACE', 1),
                                    ('CYTHON_TRACE_NOGIL', 1)]

def call_subprocess(args):
    p = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    exit_code = p.returncode
    if exit_code != 0:
        return None
    ret = output.decode().strip().split()
    if ret[0] == 'c++':
        return ret[1:]
    return ret


def get_mpi_args(mpi_executable, compile_argument, link_argument):
    compile_args = call_subprocess([mpi_executable, compile_argument])
    link_args = call_subprocess([mpi_executable, link_argument])
    if compile_args is None:
        return None
    return compile_args, link_args


# Add MPI libraries
ext_options_mpi = copy.deepcopy(ext_options)
compile_parallel = True
if RTDFLAG:
    ext_options['libraries'] = []
    ext_options['extra_link_args'] = []
    ext_options['extra_compile_args'].append('-DREADTHEDOCS')
    compile_parallel = False
else:
    try:
        import mpi4py
        _mpi4py_dir = os.path.dirname(mpi4py.__file__)
        ext_options_mpi['include_dirs'].append(_mpi4py_dir)
        # Check for existence of mpi
        ret = (
            get_mpi_args('mpic++', '-compile_info', '-link_info') or  # MPICH
            get_mpi_args('mpic++', '--showme:compile', '--showme:link')  # OpenMPI
        )
        if ret is not None:
            mpi_compile_args, mpi_link_args = ret
            ext_options_mpi['extra_compile_args'] += mpi_compile_args
            ext_options_mpi['extra_link_args'] += mpi_link_args
    except ImportError:
        compile_parallel = False


# Set coverage options in .coveragerc
if True: #COVFLAG:
    from coverage.config import HandyConfigParser
    covrc = '.coveragerc'
    cp = HandyConfigParser("")
    cp.read(covrc)
    if not cp.has_section('report'):
        cp.add_section('report')
    if cp.has_option('report', 'exclude_lines'):
        excl_list = cp.get('report', 'exclude_lines')
    else:
        excl_list = ""
    def add_excl_rule(excl_list, new_rule):
        if new_rule not in excl_list:
            excl_list += "\n" + new_rule
        return excl_list
    def rm_excl_rule(excl_list, new_rule):
        if new_rule in excl_list:
            excl_list = excl_list.replace("\n" + new_rule, "")
        return excl_list
    # Python version
    verlist = [2, 3]
    for v in verlist:
        vincl = 'pragma: Python %d' % v
        if PY_MAJOR_VERSION == v:
            excl_list = rm_excl_rule(excl_list, vincl)
        else:
            excl_list = add_excl_rule(excl_list, vincl)
    # MPI
    if compile_parallel:
        excl_list = add_excl_rule(excl_list, 'pragma: w/o MPI')
        excl_list = rm_excl_rule(excl_list, 'pragma: w/ MPI')
    else:
        excl_list = add_excl_rule(excl_list, 'pragma: w/ MPI')
        excl_list = rm_excl_rule(excl_list, 'pragma: w/o MPI')
    # Add and write
    if not cp.has_section('Cython.Coverage'):
        cp.add_section('Cython.Coverage')
    cp.set('report', 'exclude_lines', excl_list) 
    if cp.has_option('Cython.Coverage', 'exclude_lines'):
        cy_excl_list = cp.get('Cython.Coverage', 'exclude_lines')
        cy_excl_list += excl_list
    else:
        cy_excl_list = excl_list
    cp.set('Cython.Coverage', 'exclude_lines', cy_excl_list)
    with open(covrc, 'w') as fd:
        cp.write(fd)

# Cythonize modules
ext_modules = []

def make_cpp(cpp_file):
    if not os.path.isfile(cpp_file):
        open(cpp_file, 'a').close()
        assert(os.path.isfile(cpp_file))

make_cpp("cykdtree/c_kdtree.cpp")
make_cpp("cykdtree/c_utils.cpp")
if compile_parallel:
    make_cpp("cykdtree/c_parallel_utils.cpp")
    make_cpp("cykdtree/c_parallel_kdtree.cpp")

ext_modules += [
    Extension("cykdtree.kdtree",
              sources=["cykdtree/kdtree.pyx",
                       "cykdtree/c_kdtree.cpp",
                       "cykdtree/c_utils.cpp"],
              **ext_options),
    Extension("cykdtree.utils",
              sources=["cykdtree/utils.pyx",
                       "cykdtree/c_utils.cpp"],
              **ext_options)]
if compile_parallel:
    ext_modules += [
        Extension("cykdtree.parallel_utils",
                  sources=["cykdtree/parallel_utils.pyx",
                           "cykdtree/c_parallel_utils.cpp",
                           "cykdtree/c_utils.cpp"],
                  **ext_options_mpi),
        Extension("cykdtree.parallel_kdtree",
                  sources=["cykdtree/parallel_kdtree.pyx",
                           "cykdtree/c_parallel_kdtree.cpp",
                           "cykdtree/c_kdtree.cpp",
                           "cykdtree/c_utils.cpp",
                           "cykdtree/c_parallel_utils.cpp"],
                  **ext_options_mpi)
        ]
    print("compiling parallel")

class sdist(_sdist):
    # subclass setuptools source distribution builder to ensure cython
    # generated C files are included in source distribution and readme
    # is converted from markdown to restructured text.  See
    # http://stackoverflow.com/a/18418524/1382869
    def run(self):
        # Make sure the compiled Cython files in the distribution are
        # up-to-date

        try:
            import pypandoc
        except ImportError:
            raise RuntimeError(
                'Trying to create a source distribution without pypandoc. '
                'The readme will not render correctly on pypi without '
                'pypandoc so we are exiting.'
            )
        from Cython.Build import cythonize
        cythonize(ext_modules, **cyt_options)
        _sdist.run(self)

try:
    import pypandoc
    long_description = pypandoc.convert_file('README.md', 'rst')
except (ImportError, IOError):
    with open('README.md') as file:
        long_description = file.read()

setup(name='cykdtree',
      packages=find_packages(),
      include_package_data=True,
      version='0.2.5.dev0',
      description='Cython based KD-Tree',
      long_description=long_description,
      author='Meagan Lang',
      author_email='langmm.astro@gmail.com',
      url='https://langmm@bitbucket.org/langmm/cykdtree',
      keywords=['domain decomposition', 'decomposition', 'kdtree'],
      classifiers=["Programming Language :: Python",
                   "Programming Language :: C++",
                   "Operating System :: OS Independent",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: BSD License",
                   "Natural Language :: English",
                   "Topic :: Scientific/Engineering",
                   "Topic :: Scientific/Engineering :: Astronomy",
                   "Topic :: Scientific/Engineering :: Mathematics",
                   "Topic :: Scientific/Engineering :: Physics",
                   "Development Status :: 3 - Alpha"],
      license='BSD',
      zip_safe=False,
      cmdclass={'build_ext': build_ext, 'sdist': sdist},
      ext_modules=cythonize(ext_modules, **cyt_options))

