from __future__ import absolute_import, division, print_function
import numpy as np
import ctypes
from os.path import split, abspath, join


PATH = split(abspath(__file__))[0]


costs_lib = ctypes.cdll.LoadLibrary(join(PATH, 'dtw.so'))

ndmat_f_type = np.ctypeslib.ndpointer(
    dtype=np.float32, ndim=2, flags='C_CONTIGUOUS')
ndmat_i_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=2, flags='C_CONTIGUOUS')


dtw_chi_square_cpp = costs_lib.weighted_chi_square
dtw_weighted_euclidean_cpp = costs_lib.weighted_euclidean

dtw_chi_square_cpp.argtypes = [
    ndmat_f_type, ndmat_f_type, ndmat_f_type,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ndmat_f_type
]

dtw_weighted_euclidean_cpp.argtypes = [
    ndmat_f_type, ndmat_f_type, ndmat_f_type,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ndmat_f_type
]


def dtw_weighted_chi_square(qcurv, dcurv, weights, window):
    assert qcurv.dtype == np.float32, 'qcurv.dtype = %s' % qcurv.dtype
    assert dcurv.dtype == np.float32, 'dcurv.dtype = %s' % dcurv.dtype
    assert weights.dtype == np.float32, 'weights.dtype = %s' % weights.dtype
    assert qcurv.flags.c_contiguous
    assert dcurv.flags.c_contiguous
    assert weights.flags.c_contiguous
    assert qcurv.shape == dcurv.shape
    assert qcurv.shape[0] == weights.shape[0]
    assert qcurv.ndim == dcurv.ndim == weights.ndim == 2

    m, n = qcurv.shape
    costs_out = np.full((m, m), np.inf, dtype=np.float32)
    costs_out[0, 0] = 0.
    dtw_chi_square_cpp(
        qcurv, dcurv, weights, m, n, window,
        costs_out
    )

    return costs_out[-1, -1]


def dtw_weighted_euclidean(qcurv, dcurv, weights, window):
    assert qcurv.dtype == np.float32, 'qcurv.dtype = %s' % qcurv.dtype
    assert dcurv.dtype == np.float32, 'dcurv.dtype = %s' % dcurv.dtype
    assert weights.dtype == np.float32, 'weights.dtype = %s' % weights.dtype
    assert qcurv.flags.c_contiguous
    assert dcurv.flags.c_contiguous
    assert weights.flags.c_contiguous
    assert qcurv.shape == dcurv.shape
    assert qcurv.shape[0] == weights.shape[0]
    assert qcurv.ndim == dcurv.ndim == weights.ndim == 2

    m, n = qcurv.shape
    costs_out = np.full((m, m), np.inf, dtype=np.float32)
    costs_out[0, 0] = 0.
    dtw_weighted_euclidean_cpp(
        qcurv, dcurv, weights, m, n, window,
        costs_out
    )

    return costs_out[-1, -1]
