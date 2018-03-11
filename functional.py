import affine
import cv2
import dorsal_utils
import imutils
import numpy as np
from itertools import combinations
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d


def preprocess_image(img, flip, height, width):
    if flip:
        img = img[:, ::-1]

    mask = np.full(img.shape[0:2], 255, dtype=np.uint8)
    resz, M = imutils.center_pad_with_transform(img, height, width)
    mask = cv2.warpAffine(mask, M[:2], (width, height), flags=cv2.INTER_AREA)

    return resz, mask, M


def refine_localization(img, flip, pre_xform, loc_xform, scale, height, width):
    if flip:
        img = img[:, ::-1]
    msk = np.full(img.shape[0:2], 255, dtype=np.uint8)

    img_refn, msk_refn = imutils.refine_localization(
        img, msk, pre_xform, loc_xform, scale, height, width
    )

    return img_refn, msk_refn


# start, end: (i_0, j_0), (i_n, j_n)
def find_keypoints(method, segm, mask):
    segm = segm[:, :, 0]

    # use the mask to zero out regions of the response not corresponding to the
    # original image
    probs = np.zeros(segm.shape[0:2], dtype=np.float32)
    probs[mask > 0] = segm[mask > 0]
    start, end = method(probs)

    return start, end


def extract_outline(img, mask, segm, scale,
                    start, end, cost_func, allow_diagonal):
    Mscale = affine.build_scale_matrix(scale)
    points_orig = np.vstack((start, end))[:, ::-1]  # ij -> xy
    points_refn = affine.transform_points(Mscale, points_orig)

    # points are ij
    start_refn, end_refn = np.floor(points_refn[:, ::-1]).astype(np.int32)
    outline = dorsal_utils.extract_outline(
        img, mask, segm, cost_func, start_refn, end_refn, allow_diagonal
    )

    return outline


def separate_edges(method, outline):
    idx = method(outline)
    if idx is not None:
        return outline[:idx], outline[idx:]
    else:
        return None, None


# For humpback whales, set transpose_dims = True for positive curvature.
def compute_curvature(contour, scales, transpose_dims):
    # Contour is ij but compute_curvature expects xy.
    contour = contour[::-1] if transpose_dims else contour[:, ::-1]

    # Scales are chosen dynamically based on the variation of the i-dimension.
    radii = scales * (contour[:, 1].max() - contour[:, 1].min())
    curv = dorsal_utils.oriented_curvature(contour, radii)

    return curv


def compute_curvature_descriptors(curv, curv_length, scales,
                                  num_keypoints, uniform, feat_dim):
    if curv.shape[0] == curv_length:
        resampled = curv
    else:
        resampled = dorsal_utils.resampleNd(curv, curv_length)

    feat_mats = []
    for sidx, scale in enumerate(scales):
        # keypoints are at uniform intervals along contour
        if uniform:
            keypts = np.linspace(
                0, resampled.shape[0], num_keypoints, dtype=np.int32
            )
        # keypoints are the local maxima at each scale
        else:
            smoothed = gaussian_filter1d(resampled[:, sidx], 5.0)
            try:
                maxima_idx, = argrelextrema(
                    smoothed, np.greater, order=3
                )
            except ValueError:
                maxima_idx = np.array([], dtype=np.int32)
            try:
                minima_idx, = argrelextrema(
                    smoothed, np.less, order=3
                )
            except ValueError:
                minima_idx = np.array([], dtype=np.int32)

            extrema_idx = np.sort(np.hstack((minima_idx, maxima_idx)))
            # add a dummy index for comparing against the last idx
            dummy_idx = np.sort(np.hstack((
                extrema_idx, np.array([smoothed.shape[0] + 1]))
            ))
            valid_idx = (dummy_idx[1:] - dummy_idx[:-1] > 1)
            extrema_idx = extrema_idx[valid_idx]
            # sort based on distance to curv of 0.5 (straight line)
            extrema = abs(0.5 - smoothed[extrema_idx])
            sorted_idx = np.argsort(extrema)[::-1]
            # leave two spots for the start and endpoints
            extrema_idx =  extrema_idx[sorted_idx][0:num_keypoints - 2]

            sorted_extrema_idx = np.sort(extrema_idx)
            if sorted_extrema_idx.shape[0] > 0:
                if sorted_extrema_idx[0] in (0, 1):
                    sorted_extrema_idx = sorted_extrema_idx[1:]
            keypts = np.zeros(
                min(num_keypoints, 2 + sorted_extrema_idx.shape[0]),
                dtype=np.int32
            )
            keypts[0], keypts[-1] = 0, resampled.shape[0]
            keypts[1:-1] = sorted_extrema_idx

        endpts = list(combinations(keypts, 2))
        # each entry stores the features for one scale
        descriptors = np.empty((len(endpts), feat_dim), dtype=np.float32)
        for i, (idx0, idx1) in enumerate(endpts):
            subcurv = resampled[idx0:idx1, sidx]

            feat = dorsal_utils.resample(subcurv, feat_dim)

            # l2-norm across the feature dimension
            #feat /= np.sqrt(np.sum(feat * feat, axis=0))
            feat /= np.linalg.norm(feat)
            assert feat.shape[0] == feat_dim, (
                'f.shape[0] = %d != feat_dim' % (feat.shape[0], feat_dim))
            feat_norm = np.linalg.norm(feat, axis=0)
            assert np.allclose(feat_norm, 1.), (
                'norm(feat) = %.6f' % (feat_norm))

            descriptors[i] = feat
        feat_mats.append(descriptors)

    return feat_mats
