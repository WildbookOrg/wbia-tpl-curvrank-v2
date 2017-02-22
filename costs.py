import numpy as np
from functools import partial

from pydtw import dtw_weighted_euclidean
from pydtw import dtw_weighted_chi_square


def get_cost_func_dict():
    # map cost func names to function objects
    return {
        'dtw-l2': get_dtw_l2,
        'dtw-chi2': get_dtw_chi2,
        'norm-l2': get_norm_l2,
        'random': get_random,
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
