import affine
import cv2
import dorsal_utils
import imutils
import numpy as np


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
