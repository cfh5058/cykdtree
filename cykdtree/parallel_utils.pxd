cimport numpy as np
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t
from mpi4py.libmpi cimport MPI_Comm

cdef extern from "c_parallel_utils.hpp":
    ctypedef struct exch_rec:
        int src
        int dst
        uint32_t split_dim
        double split_val
        int64_t split_idx
        uint64_t left_idx
        uint64_t npts

    uint64_t parallel_distribute(double **pts, uint64_t **idx,
                                 uint64_t npts, uint32_t ndim) nogil
    uint64_t parallel_distribute(double **pts, uint64_t **idx,
                                 uint64_t npts, uint32_t ndim,
                                 MPI_Comm comm) nogil
    double parallel_pivot_value(double *pts, uint64_t *idx,
                                uint32_t ndim, uint32_t d,
                                int64_t l, int64_t r) nogil
    double parallel_pivot_value(double *pts, uint64_t *idx,
                                uint32_t ndim, uint32_t d,
                                int64_t l, int64_t r,
                                MPI_Comm comm) nogil
    int64_t parallel_select(double *pts, uint64_t *idx,
                            uint32_t ndim, uint32_t d,
                            int64_t l, int64_t r, int64_t n,
                            double &pivot_val) nogil
    int64_t parallel_select(double *pts, uint64_t *idx,
                            uint32_t ndim, uint32_t d,
                            int64_t l, int64_t r, int64_t n,
                            double &pivot_val,
                            MPI_Comm comm) nogil
    uint32_t parallel_split(double *all_pts, uint64_t *all_idx,
                            uint64_t Lidx, uint64_t n, uint32_t ndim,
                            double *mins, double *maxs,
                            int64_t &split_idx, double &split_val) nogil
    uint32_t parallel_split(double *all_pts, uint64_t *all_idx,
                            uint64_t Lidx, uint64_t n, uint32_t ndim,
                            double *mins, double *maxs,
                            int64_t &split_idx, double &split_val,
                            MPI_Comm comm) nogil
    uint64_t redistribute_split(double **all_pts, uint64_t **all_idx,
                                uint64_t npts, uint32_t ndim,
                                double *mins, double *maxs,
                                int64_t &split_idx, uint32_t &split_dim,
                                double &split_val) nogil
    uint64_t redistribute_split(double **all_pts, uint64_t **all_idx,
                                uint64_t npts, uint32_t ndim,
                                double *mins, double *maxs,
                                int64_t &split_idx, uint32_t &split_dim,
                                double &split_val, int split_rank) nogil
    uint64_t redistribute_split(double **all_pts, uint64_t **all_idx,
                                uint64_t npts, uint32_t ndim,
                                double *mins, double *maxs,
                                int64_t &split_idx, uint32_t &split_dim,
                                double &split_val, int split_rank,
                                MPI_Comm comm) nogil
    int calc_split_rank(int size) nogil
    int calc_split_rank(int size, bool split_left) nogil
    int calc_rounds(int &src_round) nogil
    int calc_rounds(int &src_round, MPI_Comm comm) nogil
    uint64_t kdtree_parallel_distribute(double **pts, uint64_t **idx,
                                        uint64_t npts, uint32_t ndim,
                                        double *left_edge, double *right_edge,
                                        bool *periodic_left, bool *periodic_right,
                                        double *domain_mins, double *domain_maxs,
                                        exch_rec &src, vector[exch_rec] &dst) nogil
    uint64_t kdtree_parallel_distribute(double **pts, uint64_t **idx,
                                        uint64_t npts, uint32_t ndim,
                                        double *left_edge, double *right_edge,
                                        bool *periodic_left, bool *periodic_right,
                                        double *domain_mins, double *domain_maxs,
                                        exch_rec &src, vector[exch_rec] &dst,
                                        MPI_Comm comm) nogil
