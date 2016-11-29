import affine
import cv2
import numpy as np


def center_pad_with_transform(img, imsize):
    old_height, old_width = img.shape[0:2]
    s = 1. * imsize / max(old_height, old_width)
    if old_height > old_width:
        tx = int(np.round((imsize - s * old_width) / 2.))
        ty = 0
    elif old_height < old_width:
        tx = 0
        ty = int(np.round((imsize - s * old_height) / 2.))
    else:
        tx = 0
        ty = 0

    M = np.array([[s, 0., tx],
                  [0., s, ty],
                  [0., 0., 1.]])

    resz = cv2.warpAffine(img, M[:2], (imsize, imsize))

    return resz, M


def refine_segmentation(img, s):
    orig_height, orig_width = img.shape[0:2]
    out_height, out_width = (s * np.ceil((
        orig_height, orig_width))).astype(np.int32)
    T70 = affine.build_scale_matrix(s)
    T70i = cv2.invertAffineTransform(T70[:2])

    A = T70i
    segm_refined = cv2.warpAffine(
        img.astype(np.float32), A[:2], (out_width, out_height),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    )

    return segm_refined


def refine_localization(img, mask, M, L, s, imsize):
    out_height, out_width = (s * np.ceil((imsize, imsize))).astype(np.int32)

    T10 = affine.build_downsample_matrix(imsize, imsize)
    T21 = L
    T32 = affine.build_upsample_matrix(imsize, imsize)
    T43 = cv2.invertAffineTransform(M[:2])
    T70 = affine.build_scale_matrix(s)

    T70i = cv2.invertAffineTransform(T70[:2])
    A = affine.multiply_matrices([T43, T32, T21, T10, T70i])

    loc_refined = cv2.warpAffine(
        img.astype(np.float32), A[:2], (out_width, out_height),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    )

    mask_refined = cv2.warpAffine(
        mask.astype(np.float32), A[:2], (out_width, out_height),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    )

    return loc_refined, mask_refined


def test_center_pad_with_transform():
    img = np.full((384, 512), 255, dtype=np.uint8)
    resz, M = center_pad_with_transform(img, 256)
    img = np.full((512, 384), 255, dtype=np.uint8)
    resz, M = center_pad_with_transform(img, 256)
    img = np.full((512, 512), 255, dtype=np.uint8)
    resz, M = center_pad_with_transform(img, 256)
    cv2.imwrite('img.png', resz)


if __name__ == '__main__':
    test_center_pad_with_transform()
