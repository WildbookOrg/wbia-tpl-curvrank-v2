import numpy as np

import ctypes
lib = ctypes.cdll.LoadLibrary('dtw.so')

ndmat_f_type = np.ctypeslib.ndpointer(
    dtype=np.float32, ndim=2, flags='C_CONTIGUOUS')
ndmat_i_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=2, flags='C_CONTIGUOUS')

dtw_curvweighted = lib.dtw_curvweighted
dtw_curvweighted.argtypes = [ndmat_f_type, ndmat_f_type,
                             ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int,
                             ndmat_f_type, ndmat_f_type]


# https://github.com/zmjjmz/ibeis-flukematch-module/blob/b6bfd75cd45f8ad3177e34a809e1d0b739150772/flukematch_lib.cpp#L25
def compute_dtw_fast(qcurv, dcurv, weights, window=50):
    weights_nd = np.array(weights, dtype=np.float32).reshape(-1, 1)
    weights_nd = np.ascontiguousarray(weights_nd)

    if qcurv.dtype != np.float32:
        qcurv_nd = np.ascontiguousarray(qcurv, dtype=np.float32)
    else:
        qcurv_nd = qcurv
    if dcurv.dtype != np.float32:
        db_curv_nd = np.ascontiguousarray(dcurv, dtype=np.float32)
    else:
        db_curv_nd = dcurv

    dcurv_nd = dcurv
    query_len = qcurv_nd.shape[0]
    db_len = dcurv_nd.shape[0]
    distance_mat = np.full((query_len, db_len), np.inf, dtype=np.float32)
    distance_mat[0, 0] = 0
    dtw_curvweighted(
        qcurv_nd, db_curv_nd, query_len, db_len, window,
        weights_nd.shape[0], weights_nd, distance_mat)

    distance = distance_mat[-1, -1]
    return distance
