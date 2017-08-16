#!/bin/sh 
set -ex
wget https://github.com/langmm/cython/archive/coverage.tar.gz -O /tmp/cython.tar.gz
tar -xzvf /tmp/cython.tar.gz
cd cython
python setup.py install
