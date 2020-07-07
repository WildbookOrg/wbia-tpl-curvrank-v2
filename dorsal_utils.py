# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import cv2
import numpy as np
from itertools import combinations
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema

from wbia_curvrank.pyastar import astar_path


# TODO: find a better way to structure these two functions
def resample(x, length):
    interp = np.linspace(0, length, num=x.shape[0], dtype=np.float32)
    f_interp = interp1d(interp, x, kind='linear')

    resamp = f_interp(np.arange(length))

    return resamp


def resampleNd(X, length):
    Xr = np.zeros((length, X.shape[1]), dtype=np.float32)
    for j in range(X.shape[1]):
        Xr[:, j] = resample(X[:, j], length)

    return Xr


def local_max2d(X):
    assert X.ndim == 2, 'X.ndim = %d != 2' % (X.ndim)
    rows_grid = np.zeros_like(X).astype(np.bool)
    cols_grid = np.zeros_like(X).astype(np.bool)
    rows_max_idx = argrelextrema(X, np.greater, order=X.shape[0], axis=0)
    cols_max_idx = argrelextrema(X, np.greater, order=X.shape[1], axis=1)

    rows_grid[rows_max_idx], cols_grid[cols_max_idx] = 1, 1

    local_max_idx = (rows_grid & cols_grid)
    i, j = np.where(local_max_idx)

    return np.vstack((i, j)).T


def find_dorsal_keypoints(X):
    coordinates = np.mgrid[0:X.shape[0]:1, 0:X.shape[1]:1].reshape(2, -1).T
    i, j = np.round(
        np.average(coordinates, weights=X.flatten(), axis=0)
    ).astype(np.int32)

    leading, trailing = X[i:, :j], X[i:, j:]

    leading_max_idx = local_max2d(leading)
    trailing_max_idx = local_max2d(trailing)

    # TODO: hack for when we cannot find any maxima
    if leading_max_idx.shape[0] > 0:
        leading_max_idx += np.array([i, 0])
        leading_first_idx = leading_max_idx[:, 1].argmin()
        start = leading_max_idx[leading_first_idx]
    else:
        start = None

    if trailing_max_idx.shape[0] > 0:
        trailing_max_idx += np.array([i, j])
        trailing_last_idx = trailing_max_idx[:, 1].argmax()
        end = trailing_max_idx[trailing_last_idx]
    else:
        end = None

    return start, end


def find_fluke_keypoints(X):
    coordinates = np.mgrid[0:X.shape[0]:1, 0:X.shape[1]:1].reshape(2, -1).T
    i, j = np.round(
        np.average(coordinates, weights=X.flatten(), axis=0)
    ).astype(np.int32)

    leading, trailing = X[:, :j], X[:, j:]

    leading_max_idx = local_max2d(leading)
    trailing_max_idx = local_max2d(trailing)

    # TODO: hack for when we cannot find any maxima
    if leading_max_idx.shape[0] > 0:
        leading_first_idx = np.linalg.norm(
            leading_max_idx - np.array([0, 0]), axis=1
        ).argmin()
        start = leading_max_idx[leading_first_idx]
    else:
        start = None

    if trailing_max_idx.shape[0] > 0:
        trailing_max_idx += np.array([0, j])
        trailing_last_idx = np.linalg.norm(
            trailing_max_idx - np.array([0, X.shape[1] - 1]), axis=1
        ).argmin()
        end = trailing_max_idx[trailing_last_idx]
    else:
        end = None

    return start, end


def dorsal_cost_func(grad, dist):
    W = 1. / np.clip(grad * dist, 1e-5, 1.)

    return W


def fluke_cost_func(grad, dist):
    norm = cv2.normalize(
        grad * dist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
    )

    W = np.exp(5. * (1. - norm))

    return W


def extract_outline(img, msk, segm, cost_func, start, end, allow_diagonal):
    assert img.ndim == 3, 'img.dim = %d != 3' % (img.ndim)
    assert segm.ndim == 2, 'segm.ndim = %d != 2' % (segm.ndim)

    # if OpenCV is not built with TBB, cvtColor hangs when run in parallel
    # following a serial call, see:
    # https://github.com/opencv/opencv/issues/5150
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if this is the case, use this workaround that imitates cvtColor
    # (~1.91 ms vs 98.5 us), or rebuild OpenCV with TBB
    #gray = np.round(
    #    np.sum(np.array([0.114, 0.587, 0.299]) * img, axis=2)
    #).astype(np.uint8)
    ksize = 3
    grad = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize),
    )
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    msk = cv2.erode(msk, kernel=kernel, iterations=1) / 255
    # NOTE: Need to update Segmentation to match the dims of msk.
    grad[msk < 1] = 0.
    grad_norm = cv2.normalize(
        grad, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
    )

    segm_norm = segm.astype(np.float32)
    segm_norm = cv2.normalize(
        segm_norm, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
    )

    _, segm_thrs = cv2.threshold(segm_norm, 0.1, 255, cv2.THRESH_BINARY_INV)
    # to ensure sufficient overlap between segmentation and gradient images
    dist, _ = cv2.distanceTransformWithLabels(
        segm_thrs.astype(np.uint8), cv2.DIST_L2, 5
    )
    dist = np.exp(-0.1 * dist)
    dist = cv2.normalize(
        dist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
    )

    W = cost_func(grad_norm, dist)

    outline = astar_path(W, start, end, allow_diagonal=allow_diagonal)
    #outline = astar_path(W, end, start, allow_diagonal=allow_diagonal)

    return outline


def separate_leading_trailing_edges(contour):
    steps = contour.shape[0] // 2 + 1
    norm = diff_of_gauss_norm(contour, steps, m=2, s=1)
    maxima_idx, = argrelextrema(norm, np.greater, order=250)

    if maxima_idx.shape[0] > 0:
        keypt = steps // 2 + maxima_idx[norm[maxima_idx].argmax()]
    else:
        keypt = None

    return keypt


def diff_of_gauss_norm(contour, steps, m=1, s=1):
    x, y = contour[:, 0], contour[:, 1]
    g1 = gaussian(np.linspace(-2 * m * s, 2 * m * s, steps), m * s)
    g2 = gaussian(np.linspace(-2 * m * s, 2 * m * s, steps), s)

    g1x = np.convolve(x, g1, mode='valid')
    g2x = np.convolve(x, g2, mode='valid')

    g1y = np.convolve(y, g1, mode='valid')
    g2y = np.convolve(y, g2, mode='valid')

    diff_of_gauss = (g1x - g2x) ** 2 + (g1y - g2y) ** 2

    return diff_of_gauss


def gaussian(u, s):
    return 1. / np.sqrt(2. * np.pi * s * s) * np.exp(-u * u / (2. * s * s))


# contour: (n, 2) array of (x, y) points
def oriented_curvature(contour, radii):
    curvature = np.zeros((contour.shape[0], len(radii)), dtype=np.float32)
    # define the radii as a fraction of either the x or y extent
    for i, (x, y) in enumerate(contour):
        center = np.array([x, y])
        dists = ((contour - center) ** 2).sum(axis=1)
        inside = dists[:, np.newaxis] <= (radii * radii)

        for j, _ in enumerate(radii):
            curve = contour[inside[:, j]]
            # sometimes only a single point lies inside the circle
            if curve.shape[0] == 1:
                curv = 0.5
            else:
                n = curve[-1] - curve[0]
                theta = np.arctan2(n[1], n[0])

                curve_p = reorient(curve, theta, center)
                center_p = np.squeeze(reorient(center[None], theta, center))
                r0 = center_p - radii[j]
                r1 = center_p + radii[j]
                r0[0] = max(curve_p[:, 0].min(), r0[0])
                r1[0] = min(curve_p[:, 0].max(), r1[0])

                area = np.trapz(curve_p[:, 1] - r0[1], curve_p[:, 0], axis=0)
                curv = area / np.prod(r1 - r0)
            curvature[i, j] = curv

    return curvature


def diff_of_gauss_descriptor(contour, m, s, num_keypoints,
                             feat_dim, contour_length, uniform):
    if contour.shape[0] == contour_length:
        resampled = contour
    else:
        resampled = resampleNd(contour, contour_length)
    if uniform:
        keypoints = np.linspace(
            0, resampled.shape[0], num_keypoints, dtype=np.int32
        )
    else:
        steps = 1 + 4 * 8 * 2
        response = diff_of_gauss_norm(resampled, steps,  m=8, s=2)

        maxima_idx, = argrelextrema(response, np.greater, order=1)

        sorted_idx = np.argsort(response[maxima_idx])[::-1]
        maxima_idx =  maxima_idx[sorted_idx][0:num_keypoints - 2]
        maxima_idx += steps // 2

        keypoints = np.zeros(
            min(num_keypoints, 2 + maxima_idx.shape[0]), dtype=np.int32)
        keypoints[0], keypoints[-1] = 0, resampled.shape[0]
        keypoints[1:-1] = np.sort(maxima_idx)

    steps = 1 + 4 * m * s
    interp_length = feat_dim + 4 * m * s
    descriptors = []
    for (idx0, idx1) in combinations(keypoints, 2):
        x, y = resampled[idx0:idx1].T

        x_interp = np.linspace(0., interp_length, num=x.shape[0])
        fx_interp = interp1d(x_interp, x, kind='linear')

        y_interp = np.linspace(0., interp_length, num=y.shape[0])
        fy_interp = interp1d(y_interp, y, kind='linear')

        x_resamp = fx_interp(np.arange(interp_length))
        y_resamp = fy_interp(np.arange(interp_length))

        curve = np.vstack((x_resamp, y_resamp)).T
        feat = diff_of_gauss_norm(curve, steps, m=m, s=s)
        feat /= np.sqrt(np.sum(feat * feat))
        assert feat.shape[0] == feat_dim

        descriptors.append(feat)

    return np.vstack(descriptors)


def rotate(radians):
    M = np.eye(3)
    M[0, 0], M[1, 1] = np.cos(radians), np.cos(radians)
    M[0, 1], M[1, 0] = np.sin(radians), -np.sin(radians)

    return M


def reorient(points, theta, center):
    M = rotate(theta)
    points_trans = points - center
    points_aug = np.hstack((points_trans, np.ones((points.shape[0], 1))))
    points_trans = np.dot(M, points_aug.transpose())
    points_trans = points_trans.transpose()[:, :2]
    points_trans += center

    assert points_trans.ndim == 2, 'points_trans.ndim == %d != 2' % (
        points_trans.ndim)

    return points_trans


def load_curv_mat_from_h5py(target, scales, curv_length):
    # each column represents a single scale
    curv_matrix = np.empty((curv_length, len(scales)), dtype=np.float32)
    with target.open('r') as h5f:
        # load each scale separately into the curvature matrix
        for sidx, s in enumerate(scales):
            curv = h5f['%.3f' % s][:]
            if curv_length is None or curv.shape[0] == curv_length:
                curv_matrix[:, sidx] = curv
            else:
                curv_matrix[:, sidx] = resample(curv, curv_length)

    return curv_matrix


def load_descriptors_from_h5py(target, scales):
    with target.open('r') as h5f:
        descriptors_dict = {s: h5f[s][:] for s in scales}

    return descriptors_dict
