# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
from functools import partial

from wbia_curvrank.pydtw import dtw_weighted_euclidean
from wbia_curvrank.pydtw import dtw_weighted_chi_square


def get_cost_func_dict():
    # map cost func names to function objects
    return {
        'dtw-l2': get_dtw_l2,
        'dtw-chi2': get_dtw_chi2,
        'norm-l2': get_norm_l2,
        'random': get_random,
        'hist': get_hist,
    }


def get_cost_func(name, **kwargs):
    cost_func_dict = get_cost_func_dict()
    return cost_func_dict[name](**kwargs)


def get_dtw_l2(**kwargs):
    cost_func = partial(dtw_weighted_euclidean, **kwargs)
    return cost_func


def get_dtw_chi2(**kwargs):
    cost_func = partial(dtw_weighted_chi_square, **kwargs)
    return cost_func


def get_norm_l2(**kwargs):
    weights = kwargs.get('weights')
    cost_func = partial(norm_l2, weights=weights)
    return cost_func


def get_random(**kwargs):
    cost_func = random_cost
    return cost_func


# some additional cost functions
def norm_l2(qcurv, dcurv, weights):
    return np.sqrt(np.sum(weights * (qcurv - dcurv) ** 2))


def random_cost(qcurv, dcurv):
    return np.random.random()


def get_hist(**kwargs):
    cost_func = hist_intersect
    return cost_func


def hist_intersect(qcurv, dcurv):
    num_bins = 10
    qhist = np.zeros((qcurv.shape[1], num_bins), dtype=np.float32)
    dhist = np.zeros((dcurv.shape[1], num_bins), dtype=np.float32)

    for j in range(qcurv.shape[1]):
        qhist[j], _ = np.histogram(
            qcurv[:, j], bins=np.linspace(0, 1, 1 + num_bins), density=True
        )
        dhist[j], _ = np.histogram(
            dcurv[:, j], bins=np.linspace(0, 1, 1 + num_bins), density=True
        )

        qhist[j] /= qhist[j].sum()
        dhist[j] /= dhist[j].sum()

    qfeat = qhist.flatten()
    dfeat = dhist.flatten()

    return -1.0 * np.sum(np.minimum(qfeat, dfeat))
