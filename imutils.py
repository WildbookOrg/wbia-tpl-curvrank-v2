# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia_curvrank_v2 import affine
import cv2
import numpy as np


def center_pad_with_transform(img, height, width):
    old_height, old_width = img.shape[0:2]
    hs = 1.0 * old_height / height
    ws = 1.0 * old_width / width
    if hs > ws:
        s = 1.0 / hs
        tx = int(np.round(abs(width - s * old_width) / 2.0))
        ty = 0
    elif hs < ws:
        s = 1.0 / ws
        tx = 0
        ty = int(np.round(abs(height - s * old_height) / 2.0))
    else:
        s = ws
        tx = 0
        ty = 0

    M = np.array([[s, 0.0, tx], [0.0, s, ty], [0.0, 0.0, 1.0]])

    resz = cv2.warpAffine(img, M[:2], (width, height), flags=cv2.INTER_AREA)

    return resz, M


def refine_segmentation(segm, s):
    orig_height, orig_width = segm.shape[0:2]
    out_height, out_width = (s * np.ceil((orig_height, orig_width))).astype(np.int32)
    T70 = affine.build_scale_matrix(s)
    T70i = cv2.invertAffineTransform(T70[:2])

    A = T70i
    segm_refined = cv2.warpAffine(
        segm.astype(np.float32),
        A[:2],
        (out_width, out_height),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
    )

    return segm_refined


def refine_localization(img, mask, M, L, s, height, width):
    out_height, out_width = (s * np.ceil((height, width))).astype(np.int32)

    T10 = affine.build_downsample_matrix(height, width)
    T21 = L
    T32 = affine.build_upsample_matrix(height, width)
    T43 = cv2.invertAffineTransform(M[:2])
    T70 = affine.build_scale_matrix(s)

    T70i = cv2.invertAffineTransform(T70[:2])
    A = affine.multiply_matrices([T43, T32, T21, T10, T70i])

    loc_refined = cv2.warpAffine(
        img.astype(np.float32),
        A[:2],
        (out_width, out_height),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
    )

    mask_refined = cv2.warpAffine(
        mask.astype(np.float32),
        A[:2],
        (out_width, out_height),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
    )

    return loc_refined, mask_refined


def test_center_pad_with_transform():
    h, w = 128, 384
    img = np.full((384, 512), 255, dtype=np.uint8)
    resz, M = center_pad_with_transform(img, h, w)
    cv2.imwrite('img1.png', resz)
    img = np.full((512, 384), 255, dtype=np.uint8)
    resz, M = center_pad_with_transform(img, h, w)
    cv2.imwrite('img2.png', resz)
    img = np.full((512, 512), 255, dtype=np.uint8)
    resz, M = center_pad_with_transform(img, h, w)
    cv2.imwrite('img3.png', resz)


if __name__ == '__main__':
    test_center_pad_with_transform()
