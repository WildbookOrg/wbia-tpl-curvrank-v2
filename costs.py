# -*- coding: utf-8 -*-
import cv2
import numpy as np


def exp_cost_func(fine, coarse):
    cost = coarse * fine
    cost = cv2.normalize(cost, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return np.exp(5.0 * (1.0 - cost)).astype(np.float32)


def hyp_cost_func(fine, coarse):
    cost = coarse * fine
    cost = cv2.normalize(cost, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return 1.0 / np.clip(cost, 1e-4, 1.0)
