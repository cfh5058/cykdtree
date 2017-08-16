#!/bin/sh
set -e
case `uname` in
Linux)
  sudo apt-get update -q
  case $1 in
    mpich) set -x;
      sudo apt-get install -y -q mpich libmpich-dev
      ;;
    openmpi) set -x;
      sudo apt-get install -y -q openmpi-bin libopenmpi-dev
      ;;
    none)
      ;;
    *)
      echo "Unknown MPI implementation:" $1
      exit 1
      ;;
  esac
  ;;
esac
