import cv2
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
