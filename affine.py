# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np


def transform_points(M, X):
    X = np.dot(M, np.hstack((X, np.ones((X.shape[0], 1)))).T)
    X = X[:2, :].T

    return X


def build_upsample_matrix(height, width):
    U = np.array([
        [(width - 1.) / 2., 0,                 (width - 1.) / 2.],
        [0.,               (height - 1.) / 2., (height - 1.) / 2.],
        [0.,               0.,                 1.]
    ], dtype=np.float32)

    return U


def build_downsample_matrix(height, width):
    D = np.array([
        [2. / (width - 1.), 0,                  -1],
        [0.,                2. / (height - 1.), -1],
        [0.,                0.,                  1.]
    ], dtype=np.float32)

    return D


def build_scale_matrix(s):
    S = np.array([
        [s,  0., 0.],
        [0., s,  0.],
        [0., 0., 1.]
    ], dtype=np.float32)

    return S


def multiply_matrices(L):
    I = np.eye(3)
    for A in L:
        if A.shape[0] == 2:
            A = np.vstack((A, np.array([0, 0, 1])))
        I = np.dot(I, A)

    return I
