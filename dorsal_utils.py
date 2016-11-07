import networkx as nx
import numpy as np
from scipy.signal import argrelextrema


def local_max2d(X):
    rows_grid = np.zeros_like(X).astype(np.bool)
    cols_grid = np.zeros_like(X).astype(np.bool)
    rows_max_idx = argrelextrema(X, np.greater, order=X.shape[0], axis=0)
    cols_max_idx = argrelextrema(X, np.greater, order=X.shape[1], axis=1)

    rows_grid[rows_max_idx], cols_grid[cols_max_idx] = 1, 1

    local_max_idx = (rows_grid & cols_grid)
    i, j = np.where(local_max_idx)

    return np.vstack((i, j)).T


def extract_outline(X):
    coordinates = np.mgrid[0:X.shape[0]:1, 0:X.shape[1]:1].reshape(2, -1).T
    i, j = np.round(
        np.average(coordinates, weights=X.flatten(), axis=0)
    ).astype(np.int32)

    leading, trailing = X[i:, :j], X[i:, j:]

    leading_max_idx = local_max2d(leading)
    trailing_max_idx = local_max2d(trailing)

    # TODO: hack for when we cannot find any maxima
    if leading_max_idx.shape[0] == 0 or trailing_max_idx.shape[0] == 0:
        return np.array([])

    leading_max_idx += np.array([i, 0])
    trailing_max_idx += np.array([i, j])

    leading_first_idx = leading_max_idx[:, 1].argmin()
    trailing_last_idx = trailing_max_idx[:, 1].argmax()

    leading_start = leading_max_idx[leading_first_idx][::-1]
    trailing_end = trailing_max_idx[trailing_last_idx][::-1]

    height, width = X.shape
    nodes = np.arange(np.prod(X.shape))
    G = nx.Graph()
    G.add_nodes_from(nodes)

    # TODO: revisit this hack
    X[:, 0], X[:, -1], X[0, :], X[-1, :] = 0., 0., 0., 0.
    W = 1. / X
    for n in nodes:
        y, x = n / width, n % width
        edges = []
        if x < width - 1:
            weight = 0.5 * (
                W[n / width, n % width] +
                W[(n + 1) / width, (n + 1) % width])
            edges.append((n, n + 1, weight))
        if y < height - 1:
            weight = 0.5 * (
                W[n / width, n % width] +
                W[(n + width) / width, (n + width) % width])
            edges.append((n, n + width, weight))

        G.add_weighted_edges_from(edges)

    for src, dst in G.edges():
        assert 0 <= src < height * width and 0 <= dst < height * width

    assert G.number_of_nodes() == height * width

    src = width * leading_start[1] + leading_start[0]
    dst = width * trailing_end[1] + trailing_end[0]

    path = nx.shortest_path(
        G, source=src, target=dst, weight='weight'
    )

    coordinates = []
    for p in path:
        coordinates.append((p % width, p / width))

    return np.vstack(coordinates)
