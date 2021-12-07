# -*- coding: utf-8 -*-
import cv2
import networkx as nx
import numpy as np
from sklearn.utils import shuffle

# import matplotlib.pyplot as plt
import torch
import torch.utils.data as data

# import matplotlib.patches as mpl_patches

from collections import defaultdict

# from scipy.spatial.distance import directed_hausdorff
# from sklearn.svm import SVC
# from sklearn.metrics import hinge_loss
# from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

# from tqdm import tqdm

import logging

# to import pystitch, which is a .pyx (cython) file, we must first do this
import pyximport

pyximport.install()
from .stitch import stitch  # NOQA


log = logging.getLogger('sciluigi-interface')


def reorient_normals(contour, normals):
    if len(normals) == 1:
        return normals

    for i in range(1, len(normals)):
        p = contour[i] - contour[i - 1]
        n = normals[i]
        if np.cross(p, n) < 0:
            normals[i] *= -1.0

    if np.dot(normals[1], normals[0]) < 0.0:
        normals *= -1.0

    return normals


def shift_contour_points(scores, max_offset):
    m, n = scores.shape
    lookup = np.zeros((m, n + 1), dtype=np.float32)
    offsets = np.full((m, n + 1), -1, dtype=np.int32)

    for col in range(0, n + 1):
        for row in range(0, m):
            offsets_idx = np.arange(-max_offset, max_offset + 1)
            offsets_val = np.full(offsets_idx.shape[0], -np.inf, dtype=np.float32)
            for i, idx in enumerate(offsets_idx):
                # Outside the grid.  Invalid move.
                if row + idx < 0 or row + idx >= m:
                    offsets_val[i] = -np.inf
                # First column.  All previous scores are zero.
                elif col == 0:
                    offsets_val[i] = scores[row, col]
                # Last column.  "Dummy" to determine endpoint.
                elif col == n:
                    offsets_val[i] = lookup[row + idx, col - 1]
                # General case.  Best cost thus far and current move.
                else:
                    offsets_val[i] = lookup[row + idx, col - 1] + scores[row, col]

            max_idx = offsets_val.argmax()
            lookup[row, col] = offsets_val[max_idx]
            offsets[row, col] = max_idx - max_offset

    end = lookup[:, n].argmax()
    cost = lookup[end, n]
    curr = end + offsets[end, n]
    path = []
    for i in range(n - 1, -1, -1):
        path.append((i, curr))
        curr = curr + offsets[curr, i]

    path = np.vstack(path)[::-1]

    return path, cost


def control_points(probs):
    _, probs_thresh = cv2.threshold(probs, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Pad the thresholded probabilities to avoid peaks on the boundary
    # in the distance transform.
    probs_thresh = cv2.copyMakeBorder(probs_thresh, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    dist = cv2.distanceTransform(probs_thresh, cv2.DIST_L2, cv2.DIST_MASK_5)
    # plt.savefig('coarse.png', bbox_inches='tight')
    # f, ax = plt.subplots(1, 1)
    # ax.set_axis_off()
    # ax.imshow(dist, cmap=plt.cm.gray)
    # plt.savefig('dist.png', bbox_inches='tight')
    # plt.show()

    nonzero_coords = np.vstack(np.where(dist > 0)).T

    smoothed = cv2.GaussianBlur(dist, (3, 3), 0.5)
    # Compute the normal vectors at each point on the contour.
    normals, hess, grad = normals_from_hessian(smoothed, nonzero_coords)

    # Localize the maxima to subpixel accuracy and do NMS.
    peaks_ij, is_max = subpixel_loc(smoothed, normals, nonzero_coords)

    # Undo the padding.
    peaks_ij = peaks_ij[1:-1, 1:-1] - np.array([1.0, 1.0])
    normals = normals[1:-1, 1:-1]
    is_max = is_max[1:-1, 1:-1]

    return peaks_ij, normals, is_max


class PatchSet(data.Dataset):
    def __init__(self, img, contour, patch_dims, patch_size):
        super(PatchSet, self).__init__()
        self.image = img
        self.contour = contour
        self.patch_dims = patch_dims
        self.patch_size = patch_size

    def __getitem__(self, index):
        pt = self.contour[index]
        img_crop_width = int(np.round(self.patch_dims[0]))
        img_crop_height = int(np.round(self.patch_dims[1]))
        crop = cv2.getRectSubPix(self.image, (img_crop_width, img_crop_height), tuple(pt))
        crop = cv2.resize(
            crop, (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA
        )
        crop = crop.transpose(2, 0, 1) / 255.0

        return torch.FloatTensor(crop)

    def __len__(self):
        return self.contour.shape[0]


def refine_contour(img, bounding_box, contour, patch_dims, patch_size, patchnet, gpu_id):

    (x0, y0, x1, y1) = bounding_box
    use_cuda = True

    patchset = PatchSet(img, contour, patch_dims, patch_size)

    batch_size = 64
    patch_iter = DataLoader(
        patchset, shuffle=False, batch_size=batch_size, num_workers=16
    )

    # crop = img[y0:y1, x0:x1]
    # f, ax = plt.subplots(1, 1)
    # ax.imshow(crop[:, :, ::-1])
    # ax.set_axis_off()
    # for i, (x, y) in enumerate(contour):
    #    x -= x0
    #    y -= y0
    #    ax.scatter(x, y, s=5, color='red')
    #    if i % 64 == 0:
    #        w, h = int(np.round(patch_dims[0])), int(np.round(patch_dims[1]))
    #        xy = np.array([x, y]) - 0.5 * np.array([w, h])
    #        rect = mpl_patches.Rectangle(xy, w, h, linewidth=5,
    #                                     edgecolor='red', facecolor='none')
    #        ax.add_patch(rect)

    # plt.savefig('patches.png', bbox_inches='tight')
    # exit(0)
    patch_probs = np.zeros((contour.shape[0], patch_size, patch_size), dtype=np.float32)
    # print('Extracting features from patches.')
    for i, x in enumerate(patch_iter):
        if use_cuda:
            x = x.cuda(gpu_id)
        with torch.no_grad():
            _, y_hat = patchnet(x)
        p_vals = (
            y_hat.cpu()
            .numpy()
            .transpose(0, 2, 3, 1)
            .reshape(-1, patch_size, patch_size, 2)
        )

        patch_probs[i * batch_size : (i + 1) * batch_size] = p_vals[:, :, :, 1]

    print(f'Patchset len: {len(patchset)}')
    print(f'i: {i}')
    part_img = img[y0:y1, x0:x1]
    contour -= np.array([x0, y0])

    patch_width, patch_height = patch_dims
    cost = np.zeros(part_img.shape[0:2], dtype=np.float32)
    # cost = np.full(part_img.shape[0:2], 100., dtype=np.float32)
    contour = contour.astype(np.float32)
    weight = np.zeros(part_img.shape[0:2], dtype=np.int32)
    stitch(contour, patch_probs, patch_width, patch_height, cost, weight)
    stitched = np.divide(cost, weight, out=np.zeros_like(cost), where=weight > 0)
    return stitched
    # return cost
    # tqdm_iter = tqdm(range(contour.shape[0]), total=contour.shape[0])
    # for k in tqdm_iter:
    #    patch = patch_probs[k]
    #    xk, yk = contour[k]

    #    x_start = np.floor(xk - patch_dims[0] / 2.)
    #    x_end = np.ceil(xk + patch_dims[0] / 2.)
    #    y_start = np.floor(yk - patch_dims[1] / 2.)
    #    y_end = np.ceil(yk + patch_dims[1] / 2.)

    #    for y in np.arange(y_start, y_end, dtype=np.int32):
    #        for x in np.arange(x_start, x_end, dtype=np.int32):
    #            if 0 <= x < cost.shape[1] and 0 <= y < cost.shape[0]:
    #                yp = (y - y_start) * patch_size / patch_height
    #                xp = (x - x_start) * patch_size / patch_width

    #                x0, y0 = int(np.floor(xp)), int(np.floor(yp))
    #                x1, y1 = int(np.ceil(xp)), int(np.ceil(yp))

    #                dx, dy = xp - x0, yp - y0

    #                interp = 0.
    #                if 0 <= y0 < patch_size and 0 <= x0 < patch_size:
    #                    interp += (1. - dx) * (1. - dy) * patch[y0, x0]
    #                if 0 <= y1 < patch_size and 0 <= x0 < patch_size:
    #                    interp += (1. - dx) * dy * patch[y1, x0]
    #                if 0 <= y0 < patch_size and 0 <= x1 < patch_size:
    #                    interp += dx * (1. - dy) * patch[y0, x1]
    #                if 0 <= y1 < patch_size and 0 <= x1 < patch_size:
    #                    interp += dx * dy * patch[y1, x1]

    #                w = gaussian(xp, yp, patch_size / 2., patch_size / 2.,
    #                             patch_size / 4.)
    #                #cost[y, x] += w * (1. - interp)
    #                cost[y, x] -= w * interp

    # start = np.round(contour[0, ::-1]).astype(np.int32)
    # end = np.round(contour[-1, ::-1]).astype(np.int32)
    # path = pyastar.astar_path(1. + cost, start, end, allow_diagonal=True)
    # fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    # ax1.imshow(part_img[:, :, ::-1], interpolation='none')
    # ax1.scatter(contour[:, 0], contour[:, 1], s=5, color='red')
    # ax1.scatter(path[:, 1], path[:, 0], s=5, color='blue')
    # ax2.imshow(cost, cmap=plt.cm.gray, interpolation='none')
    # ax2.scatter(contour[:, 0], contour[:, 1], s=5, color='red')
    # plt.show()
    # exit(0)

    # tqdm_iter = tqdm(
    #     np.random.permutation(np.arange(contour.shape[0])), total=contour.shape[0]
    # )
    # # tqdm_iter = tqdm(np.arange(contour.shape[0]),
    # #                 total=contour.shape[0])
    # print('Plotting')
    # for i in tqdm_iter:
    #     p = patch_probs[i]
    #     # n = normals[i]
    #     c = contour[i]

    #     colors = ['blue' if i == j else 'red' for j in range(contour.shape[0])]

    #     # fig, (ax1, ax2) = plt.subplots(1, 2)
    #     # ax1.imshow(part_img[:, :, ::-1])
    #     # ax1.imshow(255 * probs, alpha=0.25, cmap=plt.cm.gray)
    #     # ax1.scatter(contour[:, 0], contour[:, 1], s=5, color=colors)
    #     xy = c - 0.5 * np.array(patch_dims)
    #     rect = mpl_patches.Rectangle(
    #         xy,
    #         patch_dims[0],
    #         patch_dims[1],
    #         linewidth=1,
    #         edgecolor='red',
    #         facecolor='none',
    #     )
    #     # ax1.add_patch(rect)

    #     # ax2.imshow(p, cmap=plt.cm.gray, interpolation='none')

    #     # plt.show()
    #     # for ax in (ax1, ax2):
    #     #    ax.clear()


def choose_longest_path(G, paths):
    max_cost = -np.inf
    max_path = ([], [])  # Need the nodes and edges.
    for nodes in paths:
        edges = list(nx.utils.pairwise(nodes))
        cost = np.sum([G[u][v]['score'] for (u, v) in edges])
        if cost >= max_cost:
            max_cost = cost
            max_path = (nodes, edges)

    return max_cost, max_path


def longest_simple_path_in_forest(G):
    F = G.copy()
    nodes = F.nodes()
    terminal_nodes = [node for node in nodes if F.degree(node) == 1]
    # TODO: can be made to run in linear time.
    best0 = {node: 0.0 for node in nodes}
    best1 = {node: 0.0 for node in nodes}
    path0 = {node: [] for node in nodes}
    path1 = {node: [] for node in nodes}

    while terminal_nodes:
        v = terminal_nodes.pop(0)
        nbrs = list(F.neighbors(v))
        if nbrs:
            u = nbrs[0]
        else:
            continue

        w = F.get_edge_data(v, u)['score']
        temp = best0[v] + w
        # New first best path to u.
        if temp >= best0[u]:
            best1[u] = best0[u]
            best0[u] = temp
            path1[u] = path0[u]
            path0[u] = v
        # New second best path to u.
        elif best0[u] > temp >= best1[u]:
            best1[u] = temp
            path1[u] = v

        F.remove_node(v)
        if F.degree(u) == 1:
            terminal_nodes.append(u)

    best = {v: best0[v] + best1[v] for v in best0}
    nodes = list(best.keys())

    start = nodes[np.argmax([best[node] for node in nodes])]
    endpoints = [start]

    # Traverse the best path.
    fwd = path0[start]
    while fwd:
        endpoints.append(fwd)
        fwd = path0[fwd]

    # Take the second best path for one edge, from there the best path.
    bwd = path1[start]
    while bwd:
        endpoints.insert(0, bwd)
        bwd = path0[bwd]

    return endpoints


def link_contours(contours, max_dist):
    G = nx.Graph()

    # f, ax = plt.subplots(1, 1)
    # Store the endpoint coordinates as (i, j) and the index of the contour
    # from which the endpoint was taken to ensure two endpoints from the same
    # contour do not link to each other via a negative edge.
    endpoints, origins = [], []
    # Add the nodes and positive edges.
    for i, c in enumerate(contours):
        # When a contour is a single point, it is represented as a single node
        # with no edge between its endpoints.
        if len(c) == 1:
            s_ij = tuple(c[0])
            endpoints.append(s_ij)
            origins.append(i)
            G.add_node(s_ij, index=i, type='start')
            # ax.scatter(s_ij[1], s_ij[0])
        # Contours with two endpoints get a positive edge between them, where
        # the weight is proportional to the length.
        else:
            score = 1.0 * len(c)
            s_ij, e_ij = tuple(c[0]), tuple(c[-1])
            endpoints.append(s_ij)
            endpoints.append(e_ij)
            origins.append(i)
            origins.append(i)
            G.add_node(s_ij, index=i, type='start')
            G.add_node(e_ij, index=i, type='end')
            G.add_edge(s_ij, e_ij, score=score)
            # ax.scatter((s_ij[1], e_ij[1]), (s_ij[0], e_ij[0]))
            # ax.plot((s_ij[1], e_ij[1]), (s_ij[0], e_ij[0]))

    endpoints = np.vstack(endpoints)
    origins = np.array(origins)

    # Add the negative edges.
    for idx, (i, j) in enumerate(endpoints):
        ij = np.array([i, j])

        # For the endpoint at (i, j), we divide the space into four quadrants
        # and find the closest endpoint in each.  An edge is added to that
        # endpoint if it lies closer than the threshold and does not come from
        # the same contour.
        dist = np.linalg.norm(ij - endpoints, axis=1)
        # The lower threshold eliminates the point itself from appearing.
        close = (0 < dist) & (dist < max_dist)

        quad1 = close & (endpoints[:, 0] <= i) & (endpoints[:, 1] > j)
        quad2 = close & (endpoints[:, 0] < i) & (endpoints[:, 1] <= j)
        quad3 = close & (endpoints[:, 0] >= i) & (endpoints[:, 1] < j)
        quad4 = close & (endpoints[:, 0] > i) & (endpoints[:, 1] >= j)

        endpoints1, origins1 = endpoints[quad1], origins[quad1]
        endpoints2, origins2 = endpoints[quad2], origins[quad2]
        endpoints3, origins3 = endpoints[quad3], origins[quad3]
        endpoints4, origins4 = endpoints[quad4], origins[quad4]

        # Edge weight is the negative Euclidean distance between the endpoints.
        if endpoints1.size > 0:
            dists_to_nbrs = np.linalg.norm(ij - endpoints1, axis=1)
            nbr_idx = dists_to_nbrs.argmin()
            if origins[idx] != origins1[nbr_idx]:
                u, v = (i, j), tuple(endpoints1[nbr_idx])
                score = -1.0 * dists_to_nbrs[nbr_idx]
                G.add_edge(u, v, score=score)
                # ax.plot((u[1], v[1]), (u[0], v[0]))

        if endpoints2.size > 0:
            dists_to_nbrs = np.linalg.norm(ij - endpoints2, axis=1)
            nbr_idx = dists_to_nbrs.argmin()
            if origins[idx] != origins2[nbr_idx]:
                u, v = (i, j), tuple(endpoints2[nbr_idx])
                score = -1.0 * dists_to_nbrs[nbr_idx]
                G.add_edge(u, v, score=score)
                # ax.plot((u[1], v[1]), (u[0], v[0]))

        if endpoints3.size > 0:
            dists_to_nbrs = np.linalg.norm(ij - endpoints3, axis=1)
            nbr_idx = dists_to_nbrs.argmin()
            if origins[idx] != origins3[nbr_idx]:
                u, v = (i, j), tuple(endpoints3[nbr_idx])
                score = -1.0 * dists_to_nbrs[nbr_idx]
                G.add_edge(u, v, score=score)
                # ax.plot((u[1], v[1]), (u[0], v[0]))

        if endpoints4.size > 0:
            dists_to_nbrs = np.linalg.norm(ij - endpoints4, axis=1)
            nbr_idx = dists_to_nbrs.argmin()
            if origins[idx] != origins4[nbr_idx]:
                u, v = (i, j), tuple(endpoints4[nbr_idx])
                score = -1.0 * dists_to_nbrs[nbr_idx]
                G.add_edge(u, v, score=score)
                # ax.plot((u[1], v[1]), (u[0], v[0]))

    # plt.show()

    # nx.draw(G, layout=nx.spring_layout(G), with_labels=True)
    # plt.savefig('G.png', bbox_inches='tight')
    # plt.clf()

    # A cut vertex or articulation point is a node that, if removed, increases
    # the number of connected components in G.
    cuts = list(nx.articulation_points(G))
    # A biconnected subgraph is a subgraph that cannot be made disconnected by
    # removing a single edge.
    subgraphs = [
        sg for sg in nx.biconnected_component_subgraphs(G) if len(sg.nodes()) > 2
    ]

    # We cannot solve the longest path problem in G because there are cycles.
    # Biconnected subgraphs contain cycles, so we brute-force the longest
    # simple path between their articulation points to turn G into a forest.

    for i, sg in enumerate(subgraphs):
        union = list(set(cuts) & set(sg))
        dummies = []
        # With no cut vertex in the subgraph, we have an isolated component.
        if len(union) == 0:
            start, end = None, None
        # When only one cut vertex is in the subgraph, we're at the end of
        # the contour.  Need to add dummy nodes to choose the real endpoint.
        elif len(union) == 1:
            dummies = ['dummy%da' % i, 'dummy%db' % i]
            sg.add_nodes_from(dummies)
            # The nodes in the subgraph that aren't cuts.
            non_union = list(set(sg.nodes()) - set(cuts))
            # Add a set of edges from all non-cut vertices to the first dummy.
            sg.add_edges_from([(dummies[0], v) for v in non_union], score=0)

            # Add an edge from the first to the second dummy.
            sg.add_edge(dummies[0], dummies[1], score=0)
            start, end = union[0], dummies[1]
        # The typical case, need to find a path from one cut vertex to the
        # other.
        elif len(union) == 2:
            start, end = union[0], union[1]
        # When there are more than two articulation points in the subgraph, we
        # cannot resolve it locally.  As a simple heuristic, we choose the two
        # articulation points with the largest incident edge.  The longest
        # incoming contours are most likely to belong to the true contour.
        else:
            start, end = None, None
            max_weight1, max_weight2 = -np.inf, -np.inf
            for node in union:
                edges = G.edges(node)
                if edges:
                    weight = np.max([G[u][v]['score'] for (u, v) in edges])
                else:
                    weight = 0.0

                if weight > max_weight1:
                    max_weight2 = max_weight1
                    end = start
                    max_weight1 = weight
                    start = node
                elif weight > max_weight2:
                    max_weight2 = weight
                    end = node

        if start is not None and end is not None:
            # Simple heuristic for when brute force search is possible.
            if len(sg.edges()) <= 50:
                paths = list(nx.all_simple_paths(sg, start, end))
                _, (new_nodes, new_edges) = choose_longest_path(sg, paths)
            # When intractable, simply use an edge across the cut vertices.
            else:
                print('Too hard to solve.')
                # If one vertex is a dummy, can't use the simple heuristic.
                # Instead, we find the node with the maximum edge and use that
                # as the endpoint.
                if len(union) == 1:
                    sg.remove_node(dummies[1])  # Remove the second dummy.
                    # All the nodes that might be last in the path.
                    nbrs = sg.neighbors(dummies[0])
                    max_weight = -np.inf
                    # Find the node with the maximum edge weight.
                    for node in nbrs:
                        weight = np.max([sg[u][v]['score'] for (u, v) in sg.edges(node)])
                        if weight > max_weight:
                            max_weight = weight
                            end = node

                # Instead of searching for the longest simple path, just add
                # an edge across the subgraph.
                G.add_edge(start, end, score=0)
                new_edges = [(start, end)]
                new_nodes = [start, end]
        # We simply remove all nodes and edges in the subgraph.
        else:
            new_nodes, new_edges = [], []

        nodes_to_remove = [node for node in sg.nodes() if node not in new_nodes]
        # Need to be careful, edge (a, b) might be (b, a) in the path.
        edges_to_remove = [
            (u, v)
            for (u, v) in sg.edges()
            if (u, v) not in new_edges and (v, u) not in new_edges
        ]
        if dummies:
            sg.remove_nodes_from(dummies)
        # Remove the edges rather than the nodes so that we can still
        # reconstruct the path from contours.
        G.remove_edges_from(edges_to_remove)
        G.remove_nodes_from(nodes_to_remove)

    # nx.draw(G, layout=nx.spring_layout(G), with_labels=True)
    # plt.savefig('F.png', bbox_inches='tight')
    # plt.clf()

    # It's possible that the graph is now empty.
    contour = []
    if G.nodes():
        endpoints = longest_simple_path_in_forest(G)
        endp_idx = 0
        while endp_idx < len(endpoints):
            single_point = False
            cur = G.nodes[endpoints[endp_idx]]
            if endp_idx + 1 < len(endpoints):
                nxt = G.nodes[endpoints[endp_idx + 1]]
                # These endpoints belong to the same contour, need to add all
                # points between them in the correct order.
                if cur['index'] == nxt['index']:
                    if cur['type'] == 'start' and nxt['type'] == 'end':
                        contour.append(contours[cur['index']])
                    elif cur['type'] == 'end' and nxt['type'] == 'start':
                        contour.append(contours[cur['index']][::-1])
                    endp_idx += 2
                # This is an isolated start/endpoint.
                else:
                    single_point = True
            # This is the final point and not part of a previous contour.
            else:
                single_point = True
            # For a single point we just add that point.  This happens when
            # the linked contour enters and leaves a contourlet from the same
            # point.
            if single_point:
                if cur['type'] == 'start':
                    contour.append(contours[cur['index']][None, 0])
                else:
                    contour.append(contours[cur['index']][None, -1])
                endp_idx += 1

    return contour


def link_points(peaks, normals, is_max):
    assert peaks.shape == normals.shape
    k = 3
    pad = k // 2
    peaks = cv2.copyMakeBorder(peaks, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    normals = cv2.copyMakeBorder(normals, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    is_max = cv2.copyMakeBorder(
        is_max.astype(np.int32), pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0
    )
    is_max = is_max.astype(np.bool)

    fwd_links = np.full(peaks.shape, -1, dtype=np.int32)
    bwd_links = np.full(peaks.shape, -1, dtype=np.int32)
    idx = np.argwhere(is_max)
    for (i, j) in idx:
        n = normals[i, j]
        pi, pj = peaks[i, j]

        i0, i1 = i - k // 2, i + k // 2 + 1
        j0, j1 = j - k // 2, j + k // 2 + 1
        max_idx = np.copy(is_max[i0:i1, j0:j1])
        max_idx[k // 2, k // 2] = False
        nbr_peaks = peaks[i0:i1, j0:j1][max_idx]
        nbr_index = np.argwhere(max_idx) + np.array([i0, j0])

        # if np.allclose(n[1], 0.):
        #    c0, c1, c2 = 0., -1., pj
        # else:
        #    m = n[0] / n[1]
        #    c0, c1, c2 = 1., -m, m * pj - pi
        # d = c0 * nbr_peaks[:, 0] + c1 * nbr_peaks[:, 1] + c2
        # assert np.all(np.sign(a) == np.sign(d)), (a, d, n)
        d = (nbr_peaks[:, 0] - pi) * n[1] - (nbr_peaks[:, 1] - pj) * n[0]
        posd, negd = d > 0, d < 0
        if np.any(posd):
            pdists = np.linalg.norm(peaks[i, j] - nbr_peaks[posd], axis=1)
            assert not np.allclose(nbr_index[posd][pdists.argmin()], (i, j))
            fwd_links[i, j] = nbr_index[posd][pdists.argmin()]
        else:
            fwd_links[i, j] = (-1, -1)

        if np.any(negd):
            ndists = np.linalg.norm(peaks[i, j] - nbr_peaks[negd], axis=1)
            bwd_links[i, j] = nbr_index[negd][ndists.argmin()]
        else:
            bwd_links[i, j] = (-1, -1)

    for (i, j) in idx:
        fi, fj = fwd_links[i, j]
        if np.all(fwd_links[fi, fj] == (i, j)) or np.all(bwd_links[fi, fj] == (i, j)):
            pass
        else:
            fwd_links[i, j] = (-1, -1)

        bi, bj = bwd_links[i, j]
        if np.all(fwd_links[bi, bj] == (i, j)) or np.all(bwd_links[bi, bj] == (i, j)):
            pass
        else:
            bwd_links[i, j] = (-1, -1)

    visited = np.ones(peaks.shape[0:2], dtype=np.bool)
    visited[idx[:, 0], idx[:, 1]] = False

    def merge(i, j):
        if i < 0 or j < 0 or visited[i, j]:
            return []
        else:
            visited[i, j] = True
            # One of these will be [] because terminal or has been visited.
            fi, fj = fwd_links[i, j]
            bi, bj = bwd_links[i, j]
            return merge(fi, fj) + merge(bi, bj) + [(i, j)]

    contours = []
    for (i, j) in idx:
        if not visited[i, j]:
            visited[i, j] = True

            ni, nj = fwd_links[i, j]
            fwd_contour = merge(ni, nj)

            ni, nj = bwd_links[i, j]
            bwd_contour = merge(ni, nj)

            # We use the same recursive function for searching both
            # directions, so need to reverse the backward one.
            contour = np.vstack(
                [x for x in fwd_contour + [(i, j)] + bwd_contour[::-1]]
            ) - np.array([pad, pad])

            contours.append(contour)

    return contours


def subpixel_loc(dist, normals, idx):
    merge_thresh = np.sqrt(2.0) / 4.0
    thetas = np.rad2deg(np.arctan2(normals[:, :, 0], normals[:, :, 1])) % 360

    subpixel = np.zeros((dist.shape[0], dist.shape[1], 2), dtype=np.float32)
    is_max = np.zeros(dist.shape[0:2], dtype=np.bool)
    idx = shuffle(idx)
    for pt_idx, (i, j) in enumerate(idx):
        n = normals[i, j]  # normal vector as (i, j)
        t = thetas[i, j]  # theta in degrees [0, 360)
        if 0 <= t < 45:
            m = n[0] / n[1]  # slope of the line
            c = i - m * j  # intercept
            i0, i1 = m * (j - 1) + c, m * (j + 1) + c
            d = np.linalg.norm((i - i0, 1.0))

            p0 = (i - i0) * dist[i, j - 1] + (i0 - i + 1) * dist[i - 1, j - 1]
            p1 = (i1 - i) * dist[i + 1, j + 1] + (i + 1 - i1) * dist[i, j + 1]
        elif 45 <= t < 90:
            m = n[0] / n[1]  # slope of the line
            c = i - m * j  # intercept
            j0, j1 = (i - 1 - c) / m, (i + 1 - c) / m
            d = np.linalg.norm((1.0, j - j0))

            p0 = (j - j0) * dist[i - 1, j - 1] + (j0 - j + 1) * dist[i - 1, j]
            p1 = (j1 - j) * dist[i + 1, j + 1] + (j + 1 - j1) * dist[i + 1, j]
        elif 90 <= t < 135:
            # Need to be careful of vertical lines!
            if np.allclose(n[1], 0.0):
                j0, j1 = j, j
            else:
                m = n[0] / n[1]  # slope of the line
                c = i - m * j  # intercept
                j0, j1 = (i - 1 - c) / m, (i + 1 - c) / m
            d = np.linalg.norm((1.0, j - j0))

            p0 = (j0 - j) * dist[i - 1, j + 1] + (j + 1 - j0) * dist[i - 1, j]
            p1 = (j - j1) * dist[i + 1, j - 1] + (j1 - j + 1) * dist[i + 1, j]
        elif 135 <= t < 180:
            m = n[0] / n[1]  # slope of the line
            c = i - (n[0] / n[1]) * j  # intercept
            i0, i1 = m * (j + 1) + c, m * (j - 1) + c
            d = np.linalg.norm((i - i0, 1.0))

            p0 = (i - i0) * dist[i - 1, j + 1] + (i0 - i + 1) * dist[i, j + 1]
            p1 = (i1 - i) * dist[i + 1, j - 1] + (i + 1 - i1) * dist[i, j - 1]
        elif 180 <= t < 225:
            m = n[0] / n[1]  # slope of the line
            c = i - m * j  # intercept
            i0, i1 = m * (j + 1) + c, m * (j - 1) + c
            d = np.linalg.norm((i0 - i, 1.0))

            p0 = (i0 - i) * dist[i + 1, j + 1] + (i + 1 - i0) * dist[i, j + 1]
            p1 = (i - i1) * dist[i - 1, j - 1] + (i1 - i + 1) * dist[i, j - 1]
        elif 225 <= t < 270:
            m = n[0] / n[1]  # slope of the line
            c = i - m * j  # intercept
            j0, j1 = (i + 1 - c) / m, (i - 1 - c) / m
            d = np.linalg.norm((1.0, j0 - j))

            p0 = (j0 - j) * dist[i + 1, j + 1] + (j + 1 - j0) * dist[i + 1, j]
            p1 = (j - j1) * dist[i - 1, j - 1] + (j1 - j + 1) * dist[i - 1, j]
        elif 270 <= t < 315:
            # Need to be careful of vertical lines!
            if np.allclose(n[1], 0.0):
                j0, j1 = j, j
            else:
                m = n[0] / n[1]  # slope of the line
                c = i - m * j  # intercept
                j0, j1 = (i + 1 - c) / m, (i - 1 - c) / m
            d = np.linalg.norm((1.0, j0 - j))

            p0 = (j - j0) * dist[i + 1, j - 1] + (j0 - j + 1) * dist[i + 1, j]
            p1 = (j1 - j) * dist[i - 1, j + 1] + (j + 1 - j1) * dist[i - 1, j]
        # Because of floating point precision 360 should be inclusive.
        elif 315 <= t <= 360:
            m = n[0] / n[1]  # slope of the line
            c = i - m * j  # intercept
            i0, i1 = m * (j - 1) + c, m * (j + 1) + c
            d = np.linalg.norm((i0 - i, 1.0))

            p0 = (i0 - i) * dist[i + 1, j - 1] + (i + 1 - i0) * dist[i, j - 1]
            p1 = (i - i1) * dist[i - 1, j - 1] + (i1 - i + 1) * dist[i, j + 1]
        else:
            assert False, '%.f' % t

        pc = dist[i, j]  # Peak of the pixel at (i, j).
        # This peak is a maximum, need to localize it so subpixel accuracy.
        if p0 <= pc >= p1:
            # f, ax = plt.subplots(1, 1)
            # ax.set_axis_off()
            # i0, j0 = 1 - np.sin(np.deg2rad(t)), 1 - np.cos(np.deg2rad(t))
            # i1, j1 = 1 + np.sin(np.deg2rad(t)), 1 + np.cos(np.deg2rad(t))
            # ax.scatter(1, 1, color='red', s=10)
            # ax.scatter(j0, i0, color='red', s=10)
            # ax.scatter(j1, i1, color='red', s=10)
            # ax.imshow(dist[i - 1:i + 2, j - 1:j + 2], cmap=plt.cm.gray)
            # ax.arrow(j0, i0, dx=j1 - j0, dy=i1 - i0, color='red')
            # plt.savefig('peaks.png', bbox_inches='tight')

            # Parabola: f(-d) = p0, f(0) = pc, f(d) = p1.  Scale to [-1, 1].
            # We are solving the system of equations:
            # c0 + c1 * x + c2 * x^2 = p0 at x = -1
            # c0 + c1 * x + c2 * x^2 = pc at x =  0
            # c0 + c1 * x + c2 * x^2 = p1 at x =  1
            # c0 = pc
            c1 = 0.5 * (p1 - p0)
            c2 = 0.5 * (p0 + p1) - pc
            assert c2 <= 0.0, '2nd derivative of parabola must be negative!'
            # Peak is where first derivative is zero.
            # TODO: check local min./max.?
            ds = -d * c1 / (2.0 * c2)

            # f, ax = plt.subplots(1, 1)
            # ax.scatter(-1, p0, color='red')
            # ax.annotate('$p_0$', (-1, p0))
            # ax.scatter(0, pc, color='red')
            # ax.annotate('$p_1$', (0, pc))
            # ax.scatter(1, p1, color='red')
            # ax.annotate('$p_2$', (1, p1))
            # x = np.linspace(-1, 1, 1000)
            # y = c0 + c1 * x + c2 * x * x
            # x0 = -c1 / (2. * c2)
            # y0 = c0 + x0 * c1 + c2 * x0 * x0
            # ax.scatter(x0, y0, color='black')
            # ax.plot(x, y, color='blue')
            # plt.savefig('para.png', bbox_inches='tight')
            # exit(0)

            # Shift the peak by ds along the normal direction.
            subpixel[i, j] = np.array([i, j]) + n * ds
            is_max[i, j] = True
        # This point is not a maximum and should be suppressed.
        else:
            subpixel[i, j] = np.array([i, j])
            is_max[i, j] = False

    # After localizing peaks to subpixel accuracy, some points may now be too
    # close together or even overlap exactly.  We need to merge these before
    # linking to avoid ambiguities in the contours.
    k = 3
    idx = np.argwhere(is_max)
    to_merge = defaultdict(list)
    for (i, j) in idx:
        i0, i1 = i - k // 2, i + k // 2 + 1
        j0, j1 = j - k // 2, j + k // 2 + 1
        max_idx = np.copy(is_max[i0:i1, j0:j1])
        max_idx[k // 2, k // 2] = False
        nbr_peaks = subpixel[i0:i1, j0:j1][max_idx]
        nbr_index = np.argwhere(max_idx) + np.array([i0, j0])

        # The distances from the peak at (i, j) to its neighbors.
        dists = np.linalg.norm(subpixel[i, j] - nbr_peaks, axis=1)
        if dists.size:
            too_close = dists < merge_thresh
            # assert too_close.sum() <= 1  # Sanity check...
            # For each point, we store the (i, j) location of the peak taht is
            # too close.  There is no need to store the reverse too.
            for ix in nbr_index[too_close]:
                ij = tuple(ix)
                if ij not in to_merge:
                    to_merge[(i, j)].append(ij)

    for (i, j) in to_merge:
        nbrs_ij = to_merge[i, j]
        # Get the subpixel peak locations to be merged.
        merge_peaks = np.vstack(
            (subpixel[i, j], [subpixel[ix, jx] for (ix, jx) in nbrs_ij])
        )
        # Get the normals to be merged and account for the sign ambiguity.
        merge_normals = np.vstack(
            (
                normals[i, j],
                [
                    normals[ix, jx]
                    if np.dot(normals[ix, jx], normals[i, j]) >= 0.0
                    else -1.0 * normals[ix, jx]
                    for (ix, jx) in nbrs_ij
                ],
            )
        )

        new_peak = merge_peaks.mean(axis=0)
        new_normal = merge_normals.mean(axis=0)

        # The first point is arbitrarily chosen as the subpixel location of
        # the merged peaks.
        new_normal = new_normal / np.linalg.norm(new_normal)
        subpixel[i, j] = new_peak
        # The points that have been merged are no longer peaks.
        for (ix, jx) in nbrs_ij:
            is_max[ix, jx] = False

    return subpixel, is_max


def normals_from_hessian(x, idx):
    # Kernels for the first and second derivatives in x and y.
    Sx = np.array([[0.0, 0.0, 0.0], [-0.5, 0, 0.5], [0.0, 0.0, 0.0]], dtype=np.float32)
    Sy = np.array([[0.0, -0.5, 0.0], [0.0, 0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32)
    Sxx = np.array([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    Syy = np.array([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    Sxy = np.array(
        [[0.25, 0.0, -0.25], [0.0, 0.0, 0.0], [-0.25, 0.0, 0.25]], dtype=np.float32
    )

    dx = cv2.filter2D(x, cv2.CV_32F, Sx)
    dy = cv2.filter2D(x, cv2.CV_32F, Sy)
    dxx = cv2.filter2D(x, cv2.CV_32F, Sxx)
    dyy = cv2.filter2D(x, cv2.CV_32F, Syy)
    dxy = cv2.filter2D(x, cv2.CV_32F, Sxy)

    normals = np.zeros((x.shape[0], x.shape[1], 2), dtype=np.float32)
    hess = np.zeros((x.shape[0], x.shape[1], 2), dtype=np.float32)
    grad = np.zeros((x.shape[0], x.shape[1], 2), dtype=np.float32)
    # f, ax = plt.subplots(1, 1)
    # ax.set_axis_off()
    # ax.imshow(x, interpolation=None,
    #          cmap=plt.cm.gray)
    for (i, j) in idx:
        # Hessian matrix at the point (i,j).
        H = np.array([[dxx[i, j], dxy[i, j]], [dxy[i, j], dyy[i, j]]])
        # Derivative vector at the point (i, j).
        D = np.array([dx[i, j], dy[i, j]])

        # Choose the eigenvector corresponding to the most negative eigenvalue.
        w, v = np.linalg.eig(H)
        hv = v[:, w.argmin()][::-1]
        # Implicit scaling and weighting by 2-norm.
        dv = D[::-1]

        # Account for the sign ambiguity.
        if np.dot(dv, hv) < 0.0:
            dv = -1.0 * dv

        # temp = np.linalg.norm(H) * hv
        # ax.arrow(j, i, dx=temp[1] / 4, dy=temp[0] / 4, color='red',
        #         linewidth=2, head_width=0.05)
        # ax.arrow(j, i, dx=dv[1] / 4, dy=dv[0] / 4, color='red',
        #         linewidth=2, head_width=0.05)

        # Approx. the normal vector at a point as the weighted combination of
        # the first and second derivatives.
        nv = dv + np.linalg.norm(H) * hv
        normals[i, j] = nv / np.linalg.norm(nv)

        # temp = normals[i, j]
        # ax.arrow(j, i, dx=temp[1] / 4, dy=temp[0] / 4, color='red',
        #         linewidth=2, head_width=0.05)
        hess[i, j] = hv
        grad[i, j] = dv

    # plt.show()

    return normals, hess, grad


if __name__ == '__main__':
    x = np.array(
        [
            [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0],
            [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0],
            [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0],
            [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0],
        ],
        dtype=np.float32,
    )
    idx = (
        np.mgrid[1 : x.shape[1] - 1 : 1, 1 : x.shape[0] - 1 : 1].reshape(2, -1).T[:, ::-1]
    )
    normals_from_hessian(x, idx)
