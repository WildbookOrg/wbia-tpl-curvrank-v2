# https://stackoverflow.com/a/47704932/1205479
#cython: language_level=3
import numpy as np
from libc.math cimport ceil, exp, floor


cdef float sqr(x):
    return x * x

cdef float gaussian(float x, float y, float x0, float y0, float sigma):
    return exp(-1. * (sqr(x - x0) + sqr(y - y0)) / (2. * sqr(sigma)))

cpdef void stitch(const float[:, :] contour, const float[:, :, :] patches,
                  const float patch_width, const float patch_height,
                  float[:, :] cost, int[:, :] weight):
    cdef int k, x, y
    cdef float xk, yk, xp, yp
    cdef int x_start, x_end, y_start, y_end
    cdef int x0, x1, y0, y1
    cdef float dx, dy
    cdef float interp
    cdef float w
    cdef int patch_size
    patch_size = patches.shape[1]
    for k in range(contour.shape[0]):
        xk, yk = contour[k]

        x_start = int(floor(xk - patch_width / 2.))
        x_end = int(ceil(xk + patch_width / 2.))
        y_start = int(floor(yk - patch_height / 2.))
        y_end = int(ceil(yk + patch_height / 2.))

        for y in range(y_start, y_end + 1):
            for x in range(x_start, x_end + 1):
                if 0 <= x < cost.shape[1] and 0 <= y < cost.shape[0]:
                    yp = (y - y_start) * patch_size / patch_height
                    xp = (x - x_start) * patch_size / patch_width

                    x0, y0 = int(floor(xp)), int(floor(yp))
                    x1, y1 = int(ceil(xp)), int(ceil(yp))

                    dx, dy = xp - x0, yp - y0

                    interp = 0.
                    if 0 <= y0 < patch_size and 0 <= x0 < patch_size:
                        interp += (1. - dx) * (1. - dy) * patches[k, y0, x0]
                    if 0 <= y1 < patch_size and 0 <= x0 < patch_size:
                        interp += (1. - dx) * dy * patches[k, y1, x0]
                    if 0 <= y0 < patch_size and 0 <= x1 < patch_size:
                        interp += dx * (1. - dy) * patches[k, y0, x1]
                    if 0 <= y1 < patch_size and 0 <= x1 < patch_size:
                        interp += dx * dy * patches[k, y1, x1]

                    w = gaussian(xp, yp, patch_size / 2., patch_size / 2.,
                                 patch_size / 4.)
                    cost[y, x] += w * interp
                    weight[y, x] += 1
