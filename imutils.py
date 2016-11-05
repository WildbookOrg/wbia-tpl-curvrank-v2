import cv2
import numpy as np


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
