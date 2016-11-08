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


def refine_localization_segmentation(img, segm, M, L, s, imsize):
    orig_height, orig_width = img.shape[0:2]
    out_height, out_width = (s * np.ceil((imsize, imsize))).astype(np.int32)

    T10 = affine.build_downsample_matrix(imsize, imsize)
    T21 = L
    T32 = affine.build_upsample_matrix(imsize, imsize)
    T43 = cv2.invertAffineTransform(M[:2])
    T70 = affine.build_scale_matrix(s)

    T45 = affine.build_upsample_matrix(orig_height, orig_width)
    T67 = affine.build_downsample_matrix(out_height, out_width)

    # these can be simplified
    T70i = cv2.invertAffineTransform(T70[:2])
    T45i = cv2.invertAffineTransform(T45[:2])
    T67i = cv2.invertAffineTransform(T67[:2])

    # because the segmentation output is 128 x 128
    T70a = affine.build_downsample_matrix(imsize / 2, imsize / 2)
    T70b = affine.build_upsample_matrix(out_height, out_width)

    # the "get pixels" matrix for the image (also T56)
    A = affine.multiply_matrices([T45i, T43, T32, T21, T10, T70i, T67i])
    # the "get pixels" matrix for the probabilities: note scaling
    B = affine.multiply_matrices([T70a, 0.5 * T70i, T70b])
    # the "get points" matrix (also T70)
    #C = T70

    loc_refined = affine._transform_affine(
        A[:2], img, out_height, out_width
    )
    segm_refined = affine._transform_affine(
        B[:2], segm[:, :, None], out_height, out_width
    )

    return loc_refined, segm_refined


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
