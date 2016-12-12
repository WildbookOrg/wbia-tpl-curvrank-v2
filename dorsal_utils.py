import cv2
import networkx as nx
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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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


def extract_contour_refined(localization, segmentation, coordinates):
    assert localization.shape[0:2] == segmentation.shape[0:2], '%r != %r' % (
        localization.shape[0:2], segmentation.shape[0:2])

    gs = cv2.cvtColor(localization, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3)
    grad_m = cv2.magnitude(grad_x, grad_y)

    grad_norm = cv2.normalize(
        grad_m, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
    )

    # convert to float or will have only 0/1 after normalization
    segmentation = segmentation.astype(np.float32)
    segm_norm = cv2.normalize(
        segmentation, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
    )

    W = 1. / (grad_norm * segm_norm)

    # the scaled lr approx. contour + neighborhood points
    coordinates_extended = set()
    d = 10
    h, w = segmentation.shape[0:2]

    # coordinates: the scaled lr approx. outline
    for (x, y) in coordinates:
        neighborhood = np.mgrid[
            x - d:x + d + 1:1, y - d:y + d + 1:1
        ].reshape(2, -1).T.astype(np.int32)
        for (a, b) in neighborhood:
            if 0 <= a < w and 0 <= b < h:
                coordinates_extended.add((a, b))

    G = nx.Graph()
    G.add_nodes_from(coordinates_extended)

    for (x, y) in coordinates_extended:
        edges = []
        if (x + 1, y) in coordinates_extended and\
                not G.has_edge((x, y), (x + 1, y)):
            weight = 0.5 * (W[y, x] + W[y, x + 1])
            edges.append((
                (x, y), (x + 1, y), weight)
            )
        if (x, y + 1) in coordinates_extended and\
                not G.has_edge((x, y), (x, y + 1)):
            weight = 0.5 * (W[y, x] + W[y + 1, x])
            edges.append((
                (x, y), (x, y + 1), weight)
            )

        G.add_weighted_edges_from(edges)

    start = tuple(coordinates[0])
    end = tuple(coordinates[-1])

    assert G.has_node(start)
    assert G.has_node(end)
    assert G.number_of_nodes() == len(coordinates_extended)

    path = nx.shortest_path(G, source=start, target=end, weight='weight')

    path = np.vstack(path)
    return path


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
    for j, s in enumerate(scales):
        r = s * (contour[:, 1].max() - contour[:, 1].min())
        for i, (x, y) in enumerate(contour):
            center = np.array([x, y])

            inside = ((contour - center) ** 2).sum(axis=1) <= (r ** 2)
            curve = np.vstack(contour[inside])

            n = curve[-1] - curve[0]
            theta = np.arctan2(n[1], n[0])

            curve_p = reorient(curve, theta, center)
            center_p = np.squeeze(reorient(center[None], theta, center))

            r0 = center_p - np.array([r, r])
            r1 = center_p + np.array([r, r])
            r0[0] = max(curve_p[:, 0].min(), r0[0])
            #r1[0] = min(curve_p[:, 0].max(), r1[0])

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
