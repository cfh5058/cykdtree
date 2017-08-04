import os
from cykdtree.tests import scaling


def test_stats_run():
    scaling.stats_run(100, 1, 2, display=True)
    f = scaling.stats_run(100, 1, 2, periodic=True, overwrite=True,
                          suppress_final_output=True)
              
    assert(os.path.isfile(f))
    os.remove(f)


def test_strong_scaling():
    f = scaling.strong_scaling(npart=100, nproc_list=[1,2], ndim_list=[2],
                               periodic=True, overwrite=True,
                               suppress_final_output=True)
    assert(os.path.isfile(f))
    os.remove(f)


def test_weak_scaling():
    f = scaling.weak_scaling(npart=100, nproc_list=[1,2], ndim_list=[2],
                             periodic=True, overwrite=True,
                             suppress_final_output=True)
    assert(os.path.isfile(f))
    os.remove(f)
