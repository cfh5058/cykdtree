import sys
import os
PY2 = (sys.version_info[0] == 2)
if PY2:  # pragma: Python 2
    import cPickle as pickle
else:  # pragma: Python 3
    import pickle


def load_from_pickle(fname):
    r"""Load a pickled object from a binary file.

    Args:
        fname (str): The file that the object should be loaded from.
    
    Returns:
        object: Python object unpickled.

    Raises:
        AssertionError: If the file does not exist.

    """
    assert(os.path.isfile(fname))
    with open(fname, 'rb') as fd:
        out = pickle.load(fd)
    return out


def dump_to_pickle(fname, obj):
    r"""Dump pickled object to a binary file.

    Args:
        fname (str): Full path to file that object should be dumped to.
        obj (object): Python object to pickle.

    Raises:
        AssertionError: If the file is not created.

    """
    with open(fname, 'wb') as fd:
        if PY2:  # pragma: Python 2
            pickle.dump(obj, fd, pickle.HIGHEST_PROTOCOL)
        else:  # pragma: Python 3
            pickle.dump(obj, fd)
    assert(os.path.isfile(fname))
