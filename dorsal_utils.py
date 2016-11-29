import cv2
import networkx as nx
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
from scipy.spatial.distance import cityblock


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
    rows_grid = np.zeros_like(X).astype(np.bool)
    cols_grid = np.zeros_like(X).astype(np.bool)
    rows_max_idx = argrelextrema(X, np.greater, order=X.shape[0], axis=0)
    cols_max_idx = argrelextrema(X, np.greater, order=X.shape[1], axis=1)

    rows_grid[rows_max_idx], cols_grid[cols_max_idx] = 1, 1

    local_max_idx = (rows_grid & cols_grid)
    i, j = np.where(local_max_idx)

    return np.vstack((i, j)).T


def extract_outline(img, segm, keyp):
    probs = keyp * segm[:, :, np.newaxis]
    probs_top, probs_start, probs_end =\
        probs[:, :, 0], probs[:, :, 1], probs[:, :, 2]
    start = np.unravel_index(probs_start.argmax(axis=None), probs_start.shape)
    top = np.unravel_index(probs_top.argmax(axis=None), probs_top.shape)
    end = np.unravel_index(probs_end.argmax(axis=None), probs_end.shape)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    grad = cv2.normalize(
        grad, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
    )
    segm = segm.astype(np.float32)
    segm = cv2.normalize(
        segm, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
    )

    W = 1. / (0.25 * grad + 0.75 * segm)

    #G = nx.grid_graph(dim=list(W.shape))
    G = nx.Graph()
    nodes = [(i, j) for i in range(W.shape[0]) for j in range(W.shape[1])]
    G.add_nodes_from(nodes)
    for (i, j) in nodes:
        edges = []
        if i + 1 < W.shape[0]:
            weight = 0.5 * (W[i, j] + W[i + 1, j])
            edges.append(((i, j), (i + 1, j), weight))
        if j + 1 < W.shape[1]:
            weight = 0.5 * (W[i, j] + W[i, j + 1])
            edges.append(((i, j), (i, j + 1), weight))
        G.add_weighted_edges_from(edges)

    leading = nx.astar_path(
        G, source=start, target=top,
        weight='weight', heuristic=cityblock
    )

    trailing = nx.astar_path(
        G, source=top, target=end,
        weight='weight', heuristic=cityblock
    )

    #return W, start, top, end
    return np.vstack(leading), np.vstack(trailing)


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
