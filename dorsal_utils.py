import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema

from pyastar import astar_path


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


def find_keypoints(X):
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


def extract_outline(img, segm, start, end):
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
    grad = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    grad_norm = cv2.normalize(
        grad, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
    )
    segm_norm = segm.astype(np.float32)
    segm_norm = cv2.normalize(
        segm_norm, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
    )

    #W = -1. * np.log(grad_norm * segm_norm)

    #W = 1. / (0.25 * grad + 0.75 * segm)
    W = 1. / (np.clip(grad_norm * segm_norm, 1e-15, 1.))

    outline = astar_path(W, start, end)

    return outline


def separate_leading_trailing_edges(contour):
    steps = contour.shape[0] / 2 + 1
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


def block_curvature(contour, scales):
    curvature = np.zeros((contour.shape[0], len(scales)), dtype=np.float32)
    for j, s in enumerate(scales):
        h = s * (contour[:, 1].max() - contour[:, 1].min())
        w = s * (contour[:, 0].max() - contour[:, 0].min())
        for i, (x, y) in enumerate(contour):
            x0, x1 = x - w / 2., x + w / 2.
            y0 = max(contour[:, 1].min(), y - h / 2.)
            y1 = min(y + h / 2., contour[:, 1].max())

            x0x1 = (contour[:, 0] >= x0)
            y0y1 = (contour[:, 1] >= y0) & (contour[:, 1] <= y1)
            curve = contour[x0x1 & y0y1]

            curve[:, 0] = np.clip(curve[:, 0], x0, x1)
            area = np.trapz(curve[:, 0] - x0, curve[:, 1], axis=0)

            curvature[i, j] = area / ((x1 - x0) * (y1 - y0))

    return curvature


def oriented_curvature(contour, scales):
    curvature = np.zeros((contour.shape[0], len(scales)), dtype=np.float32)
    radii = (contour[:, 1].max() - contour[:, 1].min()) * np.array(scales)
    for i, (x, y) in enumerate(contour):
        center = np.array([x, y])
        dists = ((contour - center) ** 2).sum(axis=1)
        inside = dists[:, np.newaxis] <= (radii * radii)

        for j, _ in enumerate(scales):
            curve = contour[inside[:, j]]

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


def load_curv_mat_from_h5py(target, scales, curv_length, normalize):
    # each column represents a single scale
    curv_matrix = np.empty((curv_length, len(scales)), dtype=np.float32)
    with target.open('r') as h5f:
        # load each scale separately into the curvature matrix
        for sidx, s in enumerate(scales):
            curv = h5f['%.3f' % s][:]
            if normalize:
                curv -= curv.mean(axis=0)
                curv /= curv.std(axis=0)
            curv_matrix[:, sidx] = resample(curv, curv_length)

    return curv_matrix
