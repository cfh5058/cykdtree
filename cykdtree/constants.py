import os


COVFLAG = bool(os.environ.get('CYKDTREE_COVERAGE', None) == 'True')
PRFFLAG = bool(os.environ.get('CYKDTREE_PROFILE', None) == 'True')
PROF_ENABLED = (COVFLAG or PRFFLAG)
