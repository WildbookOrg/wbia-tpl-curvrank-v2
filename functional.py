# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia_curvrank import affine, curv, dorsal_utils, imutils, pyastar, utils
from wbia_curvrank.costs2 import exp_cost_func as cost_func
import annoy
import cv2
import numpy as np
from itertools import combinations
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d
import tqdm
import time
import torch


def preprocess_image(img, bbox, flip, pad, width_coarse, height_coarse, width_anchor, height_anchor):
    if flip:
        img = img[:, ::-1]

    x, y, w, h = bbox
    crop, _ = utils.crop_with_padding(
            img, x, y, w, h, pad
    )

    coarse_img = cv2.resize(crop, (width_coarse, height_coarse),
                              interpolation=cv2.INTER_AREA)
    coarse_img = coarse_img.transpose(2, 0, 1) / 255.

    anchor_img = cv2.resize(crop, (width_anchor, height_anchor),
                              interpolation=cv2.INTER_AREA)
    anchor_img = anchor_img[:, :, ::-1] / 255.
    anchor_img -= np.array([0.485, 0.456, 0.406])
    anchor_img /= np.array([0.229, 0.224, 0.225])
    anchor_img = anchor_img.transpose(2, 0, 1)

    return coarse_img, anchor_img, crop


def refine_by_gradient(img):
    Sx = np.array([[ 0.,  0., 0.],
                   [-0.5, 0,  0.5],
                   [ 0.,  0., 0.]], dtype=np.float32)
    Sy = np.array([[0., -0.5, 0.],
                   [0.,  0,   0.],
                   [0.,  0.5, 0.]], dtype=np.float32)
    dx = cv2.filter2D(img, cv2.CV_32F, Sx)
    dy = cv2.filter2D(img, cv2.CV_32F, Sy)

    refined = cv2.magnitude(dx, dy)
    refined = cv2.normalize(refined, None, alpha=0, beta=255,
                            norm_type=cv2.NORM_MINMAX)

    return refined


def contour_from_anchorpoints(part_img, coarse, fine, anchor_points, trim, width_fine):
    fine = cv2.cvtColor(fine, cv2.COLOR_BGR2GRAY)

    ratio = width_fine / part_img.shape[1]
    coarse_height, coarse_width = coarse.shape[0:2]
    fine = cv2.resize(fine, (0, 0), fx=ratio, fy=ratio,
                      interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(coarse, fine.shape[0:2][::-1],
                        interpolation=cv2.INTER_AREA)
    height_fine, width_fine = fine.shape[0:2]

    fine = cv2.normalize(fine.astype(np.float32), None,
                         alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    coarse = cv2.normalize(coarse.astype(np.float32), None,
                           alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    W = cost_func(coarse, fine)

    start_xy = anchor_points['start'][0]
    end_xy = anchor_points['end'][0]
    start_xy[0] = np.clip(start_xy[0], 0, W.shape[1] - 1)
    start_xy[1] = np.clip(start_xy[1], 0, W.shape[0] - 1)
    end_xy[0] = np.clip(end_xy[0], 0, W.shape[1] - 1)
    end_xy[1] = np.clip(end_xy[1], 0, W.shape[0] - 1)
    start_ij = tuple(start_xy[::-1].astype(np.int32))
    end_ij = tuple(end_xy[::-1].astype(np.int32))
    # A* expects start and endpoints in matrix coordinates.
    # TODO: set this based on the config
    path_ij = pyastar.astar_path(W, start_ij, end_ij,
                                 allow_diagonal=True)
    # Store the contour as a list of a single array.  This is to be consistent
    # with the other algorithm, that extracts multiple contour segments.
    if trim > 0 and path_ij.shape[0] > 2 * trim:
        path_ij = path_ij[trim:-trim]
    contour = [path_ij] if path_ij.size > 0 else []

    return contour


def curvature(contour, width_fine, height_fine, scales, transpose_dims):
    radii = np.array(scales) * max(height_fine, width_fine)
    if contour:
        if transpose_dims:
            # If the start point was left, it will now be top.
            curvature = [curv.oriented_curvature(c, radii)
                         for c in contour]
        else:
            # Start point is already top, but need to convert to xy.
            curvature = [curv.oriented_curvature(c[:, ::-1], radii)
                         for c in contour]
    else:
        curvature = []

    return curvature


def curvature_descriptors(contour, curvature, scales, curv_length, feat_dim, num_keypoints):
    if contour and curvature:
        contour, curvature = utils.pad_curvature_gaps(contour,
                                                      curvature)
        contour = utils.resample2d(contour, curv_length)
        # Store the resampled contour so that the keypoints align
        # during visualization.
        data = {'keypoints': {}, 'descriptors': {}, 'contour': contour}
        curvature = utils.resample1d(curvature, curv_length)
        smoothed = gaussian_filter1d(curvature, 5.0, axis=0)

        # Returns array of shape (0, 2) if no extrema.
        maxima_idx = np.vstack(argrelextrema(
            smoothed, np.greater, axis=0, order=3
        )).T
        minima_idx = np.vstack(argrelextrema(
            smoothed, np.less, axis=0, order=3
        )).T
        extrema_idx = np.vstack((maxima_idx, minima_idx))

        for j in range(smoothed.shape[1]):
            keypts_idx = extrema_idx[extrema_idx[:, 1] == j, 0]
            # There may be no local extrema at this scale.
            if keypts_idx.size > 0:
                if keypts_idx[0] > 1:
                    keypts_idx = np.hstack((0, keypts_idx))
                if keypts_idx[-1] < smoothed.shape[0] - 2:
                    keypts_idx = np.hstack(
                        (keypts_idx, smoothed.shape[0] - 1)
                    )
                extrema_val = np.abs(smoothed[keypts_idx, j] - 0.5)
                # Ensure that the start and endpoint are included.
                extrema_val[0] = np.inf
                extrema_val[-1] = np.inf

                # Keypoints in descending order of extremum value.
                sorted_idx = np.argsort(extrema_val)[::-1]
                keypts_idx = keypts_idx[sorted_idx][0:num_keypoints]

                # The keypoints need to be in ascending order to be
                # used for slicing, i.e., x[0:5] and not x[5:0].
                keypts_idx = np.sort(keypts_idx)
                pairs_of_keypts_idx = list(combinations(keypts_idx, 2))
                keypoints = np.empty((len(pairs_of_keypts_idx), 2),
                                     dtype=np.int32)
                descriptors = np.empty((len(pairs_of_keypts_idx), feat_dim),
                                       dtype=np.float32)
                for i, (idx0, idx1) in enumerate(pairs_of_keypts_idx):
                    subcurv = curvature[idx0:idx1 + 1, j]
                    feature = utils.resample1d(subcurv, feat_dim)
                    keypoints[i] = (idx0, idx1)
                    # L2-normalization of descriptor.
                    descriptors[i] = feature / np.linalg.norm(feature)
                data['descriptors'][scales[j]] = descriptors
                data['keypoints'][scales[j]] = keypoints
            # If there are no local extrema at a particular scale.
            else:
                data['descriptors'][scales[j]] = np.empty(
                    (0, feat_dim), dtype=np.float32
                )
                data['keypoints'][scales[j]] = np.empty(
                    (0, 2), dtype=np.int32
                )
    else:
        data = {'keypoints': {}, 'descriptors': {}, 'contour': []}

    return data


def build_lnbnn_index(data, fpath, num_trees=10):
    print('Adding data to index...')
    f = data.shape[1]  # feature dimension
    index = annoy.AnnoyIndex(f, metric='euclidean')
    for i, _ in tqdm.tqdm(list(enumerate(data))):
        index.add_item(i, data[i])
    print('...done')
    print('Building indices...')
    start = time.time()
    index.build(num_trees)
    end = time.time()
    print('...done (took %r seconds' % (end - start,))
    print('Saving indices...')
    index.save(fpath)
    print('...done')
    return index


# LNBNN classification using: www.cs.ubc.ca/~lowe/papers/12mccannCVPR.pdf
# Performance is about the same using: https://arxiv.org/abs/1609.06323
def lnbnn_identify(index_fpath, k, descriptors, names, search_k=-1):
    print('Loading Annoy index...')
    fdim = descriptors.shape[1]
    index = annoy.AnnoyIndex(fdim, metric='euclidean')
    index.load(index_fpath)

    # NOTE: Names may contain duplicates.  This works, but is it confusing?
    print('Performing inference...')
    scores = {name: 0.0 for name in names}
    for data in tqdm.tqdm(list(descriptors)):
        ind, dist = index.get_nns_by_vector(
            data, k + 1, search_k=search_k, include_distances=True
        )
        # entry at k + 1 is the normalizing distance
        classes = np.array([names[idx] for idx in ind[:-1]])
        for c in np.unique(classes):
            (j,) = np.where(classes == c)
            # multiple descriptors in the top-k may belong to the
            # same class
            score = dist[j.min()] - dist[-1]
            scores[c] += score

    return scores


def dtwsw_identify(query_curvs, database_curvs, names, simfunc):
    scores = {name: 0.0 for name in names}
    for name in names:
        dcurvs = database_curvs[name]
        # mxn matrix: m query curvs, n db curvs for an individual
        S = np.zeros((len(query_curvs), len(dcurvs)), dtype=np.float32)
        for i, qcurv in enumerate(query_curvs):
            for j, dcurv in enumerate(dcurvs):
                S[i, j] = simfunc(qcurv, dcurv)

        scores[name] = S.min(axis=None)

    return scores
