import os
from nose.tools import assert_raises, eq_
from cykdtree import backwards


def test_pickle():
    test_obj = {'test': 0}
    ftest = 'test_pickle.dat'
    if os.path.isfile(ftest):  # pragma: no cover
        os.remove(ftest)
    assert_raises(AssertionError, backwards.load_from_pickle, ftest)
    backwards.dump_to_pickle(ftest, test_obj)
    res = backwards.load_from_pickle(ftest)
    eq_(res, test_obj)
    os.remove(ftest)
