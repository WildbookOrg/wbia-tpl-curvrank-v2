# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia_curvrank_v2 import algo, fcnn
from wbia_curvrank_v2 import curv, pyastar, utils
from wbia_curvrank_v2.costs import exp_cost_func, hyp_cost_func
import annoy
import cv2
import numpy as np
import torch
from itertools import combinations
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d
import tqdm
import time


def preprocess_image(img, bbox, flip, pad):
    if flip:
        img = img[:, ::-1]

    x, y, w, h = bbox
    crop, cropped_bbox = utils.crop_with_padding(img, x, y, w, h, pad)

    return img, crop, cropped_bbox


def refine_by_gradient(img):
    Sx = np.array([[0.0, 0.0, 0.0], [-0.5, 0, 0.5], [0.0, 0.0, 0.0]], dtype=np.float32)
    Sy = np.array([[0.0, -0.5, 0.0], [0.0, 0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32)
    dx = cv2.filter2D(img, cv2.CV_32F, Sx)
    dy = cv2.filter2D(img, cv2.CV_32F, Sy)

    refined = cv2.magnitude(dx, dy)
    refined = cv2.normalize(refined, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    refined = cv2.cvtColor(refined, cv2.COLOR_BGR2GRAY)

    return refined


def refine_by_network(
    images,
    cropped_images,
    cropped_bboxes,
    control_points,
    width_coarse,
    height_coarse,
    width_fine,
    height_fine,
    patch_size,
    patch_params,
    device,
):

    # gpu_id = None

    patchnet = fcnn.UNet()
    patchnet.load_state_dict(torch.load(patch_params, map_location=device))
    if torch.cuda.is_available():
        patchnet.to(device)
    patchnet.eval()

    fine_probs = []
    for img, cropped_img, cp, bbox in zip(
        images, cropped_images, control_points, cropped_bboxes
    ):
        contours = cp['contours']

        if len(contours) == 0:
            pts_xy = np.array([[0.0, 0.0]], dtype=np.float64)
        else:
            all_contour_pts_xy = np.vstack(contours)[:, ::-1]
            # Map the points onto the part image.
            height_ratio = 1.0 * cropped_img.shape[0] / height_coarse
            width_ratio = 1.0 * cropped_img.shape[1] / width_coarse
            M = np.array([[width_ratio, 0.0], [0.0, height_ratio]])
            pts_xy = cv2.transform(np.array([all_contour_pts_xy]), M)[0]
            pts_xy += np.array([bbox[0], bbox[1]])

        # Map the patch size onto the image dimensions.
        patch_dims = (
            patch_size * cropped_img.shape[1] / width_fine,
            patch_size * cropped_img.shape[0] / height_fine,
        )
        # Extract patches at contour points to get fine probabilities.
        refined = algo.refine_contour(
            img, cropped_img, bbox, pts_xy, patch_dims, patch_size, patchnet, device
        )
        refined = cv2.normalize(
            refined, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )

        fine_probs.append(refined)

    return fine_probs


def control_points(coarse):
    peaks_ij, normals, is_max = algo.control_points(coarse)
    contours = algo.link_points(peaks_ij, normals, is_max)
    subpixel_contours = [peaks_ij[contour[:, 0], contour[:, 1]] for contour in contours]
    subpixel_normals = [normals[contour[:, 0], contour[:, 1]] for contour in contours]
    data = {'contours': subpixel_contours, 'normals': subpixel_normals}
    return data


def contour_from_anchorpoints(
    part_img,
    coarse,
    fine,
    anchor_points,
    trim=0,
    width_fine=1152,
    cost_func='exp',
    **kwargs
):
    ratio = width_fine / part_img.shape[1]
    coarse_height, coarse_width = coarse.shape[0:2]
    fine = cv2.resize(fine, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(coarse, fine.shape[0:2][::-1], interpolation=cv2.INTER_AREA)
    height_fine, width_fine = fine.shape[0:2]

    fine = cv2.normalize(
        fine.astype(np.float32), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
    )
    coarse = cv2.normalize(
        coarse.astype(np.float32), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
    )

    if cost_func == 'exp':
        W = exp_cost_func(coarse, fine)
    if cost_func == 'hyp':
        W = hyp_cost_func(coarse, fine)

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
    path_ij = pyastar.astar_path(W, start_ij, end_ij, allow_diagonal=True)
    if trim > 0 and path_ij.shape[0] > 2 * trim:
        path_ij = path_ij[trim:-trim]
    contour = path_ij if path_ij.size > 0 else None

    return contour


def curvature(contour, width_fine, height_fine, scales, transpose_dims):
    radii = np.array(scales) * max(height_fine, width_fine)
    if contour is not None:
        if transpose_dims:
            # If the start point was left, it will now be top.
            curvature = curv.oriented_curvature(contour, radii)
        else:
            # Start point is already top, but need to convert to xy.
            curvature = curv.oriented_curvature(contour[:, ::-1], radii)

    else:
        curvature = None

    return curvature


def curvature_descriptors(
    contour, curvature, scales, curv_length, feat_dim, num_keypoints
):
    if contour is not None and curvature is not None:
        contour = utils.resample2d(contour, curv_length)
        # Store the resampled contour so that the keypoints align
        # during visualization.
        data = {}
        curvature = utils.resample1d(curvature, curv_length)
        smoothed = gaussian_filter1d(curvature, 5.0, axis=0)

        # Returns array of shape (0, 2) if no extrema.
        maxima_idx = np.vstack(argrelextrema(smoothed, np.greater, axis=0, order=3)).T
        minima_idx = np.vstack(argrelextrema(smoothed, np.less, axis=0, order=3)).T
        extrema_idx = np.vstack((maxima_idx, minima_idx))

        for j in range(smoothed.shape[1]):
            keypts_idx = extrema_idx[extrema_idx[:, 1] == j, 0]
            # There may be no local extrema at this scale.
            if keypts_idx.size > 0:
                if keypts_idx[0] > 1:
                    keypts_idx = np.hstack((0, keypts_idx))
                if keypts_idx[-1] < smoothed.shape[0] - 2:
                    keypts_idx = np.hstack((keypts_idx, smoothed.shape[0] - 1))
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
                descriptors = np.empty(
                    (len(pairs_of_keypts_idx), feat_dim), dtype=np.float32
                )
                for i, (idx0, idx1) in enumerate(pairs_of_keypts_idx):
                    subcurv = curvature[idx0 : idx1 + 1, j]
                    feature = utils.resample1d(subcurv, feat_dim)
                    # L2-normalization of descriptor.
                    descriptors[i] = feature / np.linalg.norm(feature)
                data[scales[j]] = descriptors
            # If there are no local extrema at a particular scale.
            else:
                data[scales[j]] = np.empty((0, feat_dim), dtype=np.float32)
        success_ = True
    else:
        data = {}
        success_ = False

    return success_, data


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
