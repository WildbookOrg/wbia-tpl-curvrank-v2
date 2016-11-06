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


def center_pad(img, length):
    old_height, old_width = img.shape[:2]
    scale = (1. * length) / max(old_height, old_width)
    rsz = cv2.resize(
        img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    new_height, new_width = rsz.shape[:2]
    if new_height > new_width:
        padded = np.zeros((length, length, 3), dtype=np.uint8)
        start = int(np.round((length - new_width) / 2.))
        end = start + new_width
        padded[:, start:end] = rsz
        return padded
    elif new_height < new_width:
        padded = np.zeros((length, length, 3), dtype=np.uint8)
        start = int(np.round((length - new_height) / 2.))
        end = start + new_height
        padded[start:end, :] = rsz
        return padded
    else:
        return rsz


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
