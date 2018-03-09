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
