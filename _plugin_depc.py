# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia.control import controller_inject  # NOQA
import cv2
import numpy as np
import utool as ut
import vtool as vt
from wbia import dtool
import wbia


_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)

register_preproc_annot = controller_inject.register_preprocs['annot']

DEFAULT_WIDTH_COARSE = {
    'fluke': 384,
    'dorsal': 256,
}
DEFAULT_HEIGHT_COARSE = {
    'fluke': 192,
    'dorsal': 256,
}
DEFAULT_WIDTH_FINE = {
    'fluke': 1152,
    'dorsal': 1024,
}
DEFAULT_HEIGHT_FINE = {
    'fluke': 576,
    'dorsal': 1024,
}
DEFAULT_WIDTH_ANCHOR = {
    'fluke': 224,
    'dorsal': 224,
}
DEFAULT_HEIGHT_ANCHOR = {
    'fluke': 224,
    'dorsal': 224,
}
DEFAULT_PATCH_SIZE = {'fluke': 128, 'dorsal': None}
DEFAULT_SCALE = {
    'fluke': 3,
    'dorsal': 4,
}
DEFAULT_SCALES = {
    'fluke': np.array([0.02, 0.04, 0.06, 0.08], dtype=np.float32),
    'dorsal': np.array([0.04, 0.06, 0.08, 0.10], dtype=np.float32),
}
DEFAULT_ALLOW_DIAGONAL = {
    'fluke': True,
    'dorsal': False,
}
DEFAULT_TRANSPOSE_DIMS = {
    'fluke': True,
    'dorsal': False,
}


INDEX_NUM_TREES = 10
INDEX_NUM_ANNOTS = 2500  # 1000
INDEX_LNBNN_K = 2
INDEX_SEARCH_D = 1  # 1
INDEX_SEARCH_K = INDEX_LNBNN_K * INDEX_NUM_TREES * INDEX_SEARCH_D
# INDEX_SEARCH_K = 10000


DEFAULT_DORSAL_TEST_CONFIG = {
    'curvrank_daily_cache': True,
    'curvrank_daily_tag': 'global',
    'curvrank_cache_recompute': False,
    'curvrank_model_type': 'dorsal',
    'curvrank_pad': 0.1,
    'curvrank_width_coarse': DEFAULT_WIDTH_COARSE['dorsal'],
    'curvrank_height_coarse': DEFAULT_HEIGHT_COARSE['dorsal'],
    'curvrank_width_fine': DEFAULT_WIDTH_FINE['dorsal'],
    'curvrank_height_fine': DEFAULT_HEIGHT_FINE['dorsal'],
    'curvrank_width_anchor': DEFAULT_WIDTH_ANCHOR['dorsal'],
    'curvrank_height_anchor': DEFAULT_HEIGHT_ANCHOR['dorsal'],
    'curvrank_patch_size': DEFAULT_PATCH_SIZE['dorsal'],
    'curvrank_trim': 0,
    'curvrank_cost_func': 'hyp',
    'curvrank_scale': DEFAULT_SCALE['dorsal'],
    'curvature_scales': DEFAULT_SCALES['dorsal'],
    'outline_allow_diagonal': DEFAULT_ALLOW_DIAGONAL['dorsal'],
    'curvature_transpose_dims': DEFAULT_TRANSPOSE_DIMS['dorsal'],
    'curvature_descriptor_curv_length': 1024,
    'curvature_descriptor_num_keypoints': 32,
    'curvature_descriptor_uniform': False,
    'curvature_descriptor_feat_dim': 32,
    'index_trees': INDEX_NUM_TREES,
    'index_search_k': INDEX_SEARCH_K,
    'index_lnbnn_k': INDEX_LNBNN_K,
}


DEFAULT_FLUKE_TEST_CONFIG = {
    'curvrank_daily_cache': True,
    'curvrank_daily_tag': 'global',
    'curvrank_cache_recompute': False,
    'curvrank_model_type': 'fluke',
    'curvrank_pad': 0.1,
    'curvrank_width_coarse': DEFAULT_WIDTH_COARSE['fluke'],
    'curvrank_height_coarse': DEFAULT_HEIGHT_COARSE['fluke'],
    'curvrank_width_fine': DEFAULT_WIDTH_FINE['fluke'],
    'curvrank_height_fine': DEFAULT_HEIGHT_FINE['fluke'],
    'curvrank_width_anchor': DEFAULT_WIDTH_ANCHOR['fluke'],
    'curvrank_height_anchor': DEFAULT_HEIGHT_ANCHOR['fluke'],
    'curvrank_patch_size': DEFAULT_PATCH_SIZE['fluke'],
    'curvrank_trim': 0,
    'curvrank_cost_func': 'exp',
    'curvrank_scale': DEFAULT_SCALE['fluke'],
    'curvature_scales': DEFAULT_SCALES['fluke'],
    'outline_allow_diagonal': DEFAULT_ALLOW_DIAGONAL['fluke'],
    'curvature_transpose_dims': DEFAULT_TRANSPOSE_DIMS['fluke'],
    'curvature_descriptor_curv_length': 1024,
    'curvature_descriptor_num_keypoints': 32,
    'curvature_descriptor_uniform': False,
    'curvature_descriptor_feat_dim': 32,
    'index_trees': INDEX_NUM_TREES,
    'index_search_k': INDEX_SEARCH_K,
    'index_lnbnn_k': INDEX_LNBNN_K,
}


DEFAULT_DEPC_KEY_MAPPING = {
    'curvrank_daily_cache': 'use_daily_cache',
    'curvrank_daily_tag': 'daily_cache_tag',
    'curvrank_cache_recompute': 'force_cache_recompute',
    'curvrank_model_type': 'model_type',
    'curvrank_pad': 'pad',
    'curvrank_width_coarse': 'width_coarse',
    'curvrank_height_coarse': 'height_coarse',
    'curvrank_width_fine': 'width_fine',
    'curvrank_height_fine': 'height_fine',
    'curvrank_width_anchor': 'width_anchor',
    'curvrank_height_anchor': 'height_anchor',
    'curvrank_patch_size': 'patch_size',
    'curvrank_trim': 'trim',
    'curvrank_cost_func': 'cost_func',
    'curvrank_scale': 'scale',
    'curvature_scales': 'scales',
    'outline_allow_diagonal': 'allow_diagonal',
    'curvature_transpose_dims': 'transpose_dims',
    'curvature_descriptor_curv_length': 'curv_length',
    'curvature_descriptor_num_keypoints': 'num_keypoints',
    'curvature_descriptor_uniform': 'uniform',
    'curvature_descriptor_feat_dim': 'feat_dim',
    'index_trees': 'num_trees',
    'index_search_k': 'search_k',
    'index_lnbnn_k': 'lnbnn_k',
}


ROOT = wbia.const.ANNOTATION_TABLE


def zip_coords(ys, xs):
    return np.array(list(zip(ys, xs)))


def get_zipped(depc, tablename, col_ids, y_key, x_key, config=None):
    if config is None:
        ys = depc.get_native(tablename, col_ids, y_key)
        xs = depc.get_native(tablename, col_ids, x_key)
    else:
        ys = depc.get(tablename, col_ids, y_key, config=config)
        xs = depc.get(tablename, col_ids, x_key, config=config)
    return zip_coords(ys, xs)


def _convert_depc_config_to_kwargs_config(config):
    config_ = {}
    for key, value in DEFAULT_DEPC_KEY_MAPPING.items():
        if key in config:
            config_[value] = config[key]
    return config_


def _convert_kwargs_config_to_depc_config(config):
    config_ = {}
    for value, key in DEFAULT_DEPC_KEY_MAPPING.items():
        if key in config:
            config_[value] = config[key]
    return config_


class PreprocessConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('curvrank_pad', 0.1),
            ut.ParamInfo('ext', '.npy', hideif='.p'),
        ]


@register_preproc_annot(
    tablename='preprocess_two',
    parents=[ROOT],
    colnames=[
        'img',
        'cropped_img',
        'cropped_bbox',
    ],
    coltypes=[
        ('extern', np.load, np.save),
        ('extern', np.load, np.save),
        np.ndarray,
    ],
    configclass=PreprocessConfig,
    fname='curvrank_v2_unoptimized',
    rm_extern_on_delete=True,
    chunksize=256,
)
def wbia_plugin_curvrank_v2_preprocessing_depc(depc, aid_list, config=None):
    r"""
    Pre-process images for CurvRank with Dependency Cache (depc)

    Args:
        depc      (Dependency Cache): IBEIS dependency cache object
        aid_list  (list of int): list of annot rowids (aids)
        config    (PreprocessConfig): config for depcache

    CommandLine:
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_preprocessing_depc
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_preprocessing_depc:0
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_preprocessing_depc:1
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_preprocessing_depc:2

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> cropped_images = ibs.depc_annot.get('preprocess_two', aid_list, 'cropped_img', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> cropped_image = cropped_images[0]
        >>> assert ut.hash_data(cropped_image) in ['zrtghjovbhnangjdlsqtfvrntlzqmaey']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> aid_list *= 10
        >>> cropped_images = ibs.depc_annot.get('preprocess_two', aid_list, 'cropped_img', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> cropped_image = cropped_images[0]
        >>> assert ut.hash_data(cropped_image) in ['zrtghjovbhnangjdlsqtfvrntlzqmaey']

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(7)
        >>> cropped_images = ibs.depc_annot.get('preprocess_two', aid_list, 'cropped_img', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> cropped_image = cropped_images[0]
        >>> assert ut.hash_data(cropped_image) in ['dhqxniyfoaufwcjasypkgkiwchiytslz']
    """
    ibs = depc.controller

    pad = config['curvrank_pad']

    images, cropped_images, cropped_bboxes = ibs.wbia_plugin_curvrank_v2_preprocessing(
        aid_list, pad
    )

    for image, cropped_image, cropped_bbox in zip(images, cropped_images, cropped_bboxes):
        yield (
            image,
            cropped_image,
            np.array(cropped_bbox),
        )


class CoarseProbabilitiesConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('curvrank_width_coarse', DEFAULT_WIDTH_COARSE['fluke']),
            ut.ParamInfo('curvrank_height_coarse', DEFAULT_HEIGHT_COARSE['fluke']),
            ut.ParamInfo('curvrank_model_type', 'fluke'),
            ut.ParamInfo('ext', '.npy', hideif='.npy'),
        ]


@register_preproc_annot(
    tablename='coarse_two',
    parents=['preprocess_two'],
    colnames=['coarse_probabilities'],
    coltypes=[('extern', np.load, np.save)],
    configclass=CoarseProbabilitiesConfig,
    fname='curvrank_v2_unoptimized',
    rm_extern_on_delete=True,
    chunksize=128,
)
# chunksize defines the max number of 'yield' below that will be called in a chunk
# so you would decrease chunksize on expensive calculations
def wbia_plugin_curvrank_v2_coarse_probabilities_depc(
    depc, preprocess_rowid_list, config=None
):
    r"""
    Extract coarse probabilities for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_coarse_probabilities_depc
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_coarse_probabilities_depc:0
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_coarse_probabilities_depc:1
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_coarse_probabilities_depc:2

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> coarse_probabilities = ibs.depc_annot.get('coarse_two', aid_list, 'coarse_probabilities',  config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> coarse_probability = coarse_probabilities[0]
        >>> assert ut.hash_data(coarse_probability) in ['qnusxayrvvygnvllicwgeroesouxdfkh']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> aid_list *= 10
        >>> coarse_probabilities = ibs.depc_annot.get('coarse_two', aid_list, 'coarse_probabilities',  config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> coarse_probability = coarse_probabilities[0]
        >>> assert ut.hash_data(coarse_probability) in ['qnusxayrvvygnvllicwgeroesouxdfkh']

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(7)
        >>> coarse_probabilities = ibs.depc_annot.get('coarse_two', aid_list, 'coarse_probabilities',  config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> coarse_probability = coarse_probabilities[0]
        >>> assert ut.hash_data(coarse_probability) in ['mwolbzkqaflwifvrklakgfxbyvogooog']
    """
    ibs = depc.controller

    cropped_images = depc.get_native(
        'preprocess_two', preprocess_rowid_list, 'cropped_img'
    )

    config_ = _convert_depc_config_to_kwargs_config(config)

    coarse_probabilities = ibs.wbia_plugin_curvrank_v2_coarse_probabilities(
        cropped_images, **config_
    )

    for coarse_prob in coarse_probabilities:
        yield (coarse_prob,)


class FineProbabilitiesConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('curvrank_width_coarse', DEFAULT_WIDTH_COARSE['fluke']),
            ut.ParamInfo('curvrank_height_coarse', DEFAULT_HEIGHT_COARSE['fluke']),
            ut.ParamInfo('curvrank_width_fine', DEFAULT_WIDTH_FINE['fluke']),
            ut.ParamInfo('curvrank_height_fine', DEFAULT_HEIGHT_FINE['fluke']),
            ut.ParamInfo('curvrank_patch_site', DEFAULT_PATCH_SIZE['fluke']),
            ut.ParamInfo('curvrank_model_type', 'fluke'),
            ut.ParamInfo('ext', '.npy', hideif='.npy'),
        ]


@register_preproc_annot(
    tablename='fine_two',
    parents=['preprocess_two', 'coarse_two'],
    colnames=[
        'fine_img',
        'width',
        'height',
    ],
    coltypes=[
        ('extern', np.load, np.save),
        int,
        int,
    ],
    configclass=FineProbabilitiesConfig,
    fname='curvrank_v2_unoptimized',
    rm_extern_on_delete=True,
    chunksize=256,
)
# chunksize defines the max number of 'yield' below that will be called in a chunk
# so you would decrease chunksize on expensive calculations
def wbia_plugin_curvrank_v2_fine_probabilities_depc(
    depc, preprocess_rowid_list, coarse_rowid_list, config=None
):
    r"""
    Extract fine probabilities for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_fine_probabilities_depc
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_fine_probabilities_depc:0
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_fine_probabilities_depc:1
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_fine_probabilities_depc:2

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> fine_probabilities = ibs.depc_annot.get('fine_two', aid_list, 'fine_img', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> fine_probability = fine_probabilities[0]
        >>> assert ut.hash_data(fine_probability) in ['vnlujxwbtwejjmvmsqwitopeoqejchdm']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> aid_list *= 10
        >>> fine_probabilities = ibs.depc_annot.get('fine_two', aid_list, 'fine_img', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> fine_probability = fine_probabilities[0]
        >>> assert ut.hash_data(fine_probability) in ['vnlujxwbtwejjmvmsqwitopeoqejchdm']

    Example2:
        >>> # DISABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(7)
        >>> fine_probabilities = ibs.depc_annot.get('fine_two', aid_list, 'fine_img', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> fine_probability = fine_probabilities[0]
        >>> assert ut.hash_data(fine_probability) in ['frhhbeoukfgsztkcutnnznnjdrjyxmkc', 'tknfmnvyakedytrpfqcirjkmfakirqgs', 'yvlacxoolxgdfpmjboymovhcgjdjhjqc', 'giamwnlbinzynmjckvqtrxgkzbvhlqnr']
    """
    ibs = depc.controller

    imgs = depc.get_native('preprocess_two', preprocess_rowid_list, 'img')
    cropped_imgs = depc.get_native('preprocess_two', preprocess_rowid_list, 'cropped_img')
    cropped_bboxes = depc.get_native(
        'preprocess_two', preprocess_rowid_list, 'cropped_bbox'
    )

    coarse_probabilities = depc.get_native(
        'coarse_two', coarse_rowid_list, 'coarse_probabilities'
    )

    config_ = _convert_depc_config_to_kwargs_config(config)

    fine_probabilities = ibs.wbia_plugin_curvrank_v2_fine_probabilities(
        imgs, cropped_imgs, cropped_bboxes, coarse_probabilities, **config_
    )

    for fine_prob in fine_probabilities:
        (
            width,
            height,
        ) = fine_prob.shape[:2]
        yield (
            fine_prob,
            width,
            height,
        )


class AnchorPointsConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('curvrank_width_fine', DEFAULT_WIDTH_FINE['fluke']),
            ut.ParamInfo('curvrank_width_anchor', DEFAULT_WIDTH_ANCHOR['fluke']),
            ut.ParamInfo('curvrank_height_anchor', DEFAULT_HEIGHT_ANCHOR['fluke']),
            ut.ParamInfo('curvrank_model_type', 'fluke'),
        ]


@register_preproc_annot(
    tablename='anchor_two',
    parents=['preprocess_two'],
    colnames=[
        'start',
        'end',
    ],
    coltypes=[
        np.ndarray,
        np.ndarray,
    ],
    configclass=AnchorPointsConfig,
    fname='curvrank_v2_unoptimized',
    rm_extern_on_delete=True,
    chunksize=128,
)
# chunksize defines the max number of 'yield' below that will be called in a chunk
# so you would decrease chunksize on expensive calculations
def wbia_plugin_curvrank_v2_anchor_points_depc(
    depc,
    preprocess_rowid_list,
    config=None,
):
    r"""
    Anchor Points for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_anchor_points_depc
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_anchor_points_depc:0
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_anchor_points_depc:1
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_anchor_points_depc:2

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> start = ibs.depc_annot.get('anchor_two', aid_list, 'start', config=DEFAULT_FLUKE_TEST_CONFIG)[0]
        >>> end = ibs.depc_annot.get('anchor_two', aid_list, 'end', config=DEFAULT_FLUKE_TEST_CONFIG)[0]
        >>> start = np.around(start, 2).tolist()
        >>> end = np.around(end, 2).tolist()
        >>> assert start == [[24.73, 44.67]]
        >>> assert end == [[1073.62, 24.86]]

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> aid_list *= 10
        >>> start = ibs.depc_annot.get('anchor_two', aid_list, 'start', config=DEFAULT_FLUKE_TEST_CONFIG)[0]
        >>> end = ibs.depc_annot.get('anchor_two', aid_list, 'end', config=DEFAULT_FLUKE_TEST_CONFIG)[0]
        >>> start = np.around(start, 2).tolist()
        >>> end = np.around(end, 2).tolist()
        >>> assert start == [[24.73, 44.67]]
        >>> assert end == [[1073.62, 24.86]]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(7)
        >>> start = ibs.depc_annot.get('anchor_two', aid_list, 'start', config=DEFAULT_DORSAL_TEST_CONFIG)[0]
        >>> end = ibs.depc_annot.get('anchor_two', aid_list, 'end', config=DEFAULT_DORSAL_TEST_CONFIG)[0]
        >>> start = np.around(start, 2).tolist()
        >>> end = np.around(end, 2).tolist()
        >>> assert start == [[52.65, 553.15]]
        >>> assert end == [[868.04, 558.56]]
    """
    ibs = depc.controller

    cropped_images = depc.get_native(
        'preprocess_two', preprocess_rowid_list, 'cropped_img'
    )

    config_ = _convert_depc_config_to_kwargs_config(config)

    anchor_points = ibs.wbia_plugin_curvrank_v2_anchor_points(cropped_images, **config_)

    for pt in anchor_points:
        start = pt['start']
        end = pt['end']
        yield (
            start,
            end,
        )


class ContoursConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('curvrank_trim', 0),
            ut.ParamInfo('curvrank_width_fine', DEFAULT_WIDTH_FINE['fluke']),
            ut.ParamInfo('curvrank_cost_func', 'exp'),
        ]


@register_preproc_annot(
    tablename='contour_two',
    parents=['coarse_two', 'fine_two', 'anchor_two', 'preprocess_two'],
    colnames=['contour'],
    coltypes=[np.ndarray],
    configclass=ContoursConfig,
    fname='curvrank_v2_unoptimized',
    rm_extern_on_delete=True,
    chunksize=256,
)
# chunksize defines the max number of 'yield' below that will be called in a chunk
# so you would decrease chunksize on expensive calculations
def wbia_plugin_curvrank_v2_contours_depc(
    depc,
    anchor_rowid_list,
    fine_rowid_list,
    coarse_rowid_list,
    preprocess_rowid_list,
    config=None,
):
    r"""
    Contours for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_contours_depc
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_contours_depc:0
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_contours_depc:1
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_contours_depc:2

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> contours = ibs.depc_annot.get('contour_two', aid_list, None, config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> contour = contours[0][0]
        >>> assert ut.hash_data(contour) in ['jluhaoxjgacguizqrxyjvpglbscshlfv']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> aid_list *= 10
        >>> contours = ibs.depc_annot.get('contour_two', aid_list, None, config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> contour = contours[0][0]
        >>> assert ut.hash_data(contour) in ['jluhaoxjgacguizqrxyjvpglbscshlfv']

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(7)
        >>> contours = ibs.depc_annot.get('contour_two', aid_list, None, config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> contour = contours[0][0]
        >>> assert ut.hash_data(contour) in ['dpbtamrdhmrptmqkxkbtzgtzaheskxkj']
    """
    ibs = depc.controller

    start = depc.get_native('anchor_two', anchor_rowid_list, 'start')
    end = depc.get_native('anchor_two', anchor_rowid_list, 'end')
    anchor_points = []
    for s, e in zip(start, end):
        anchor_points.append({'start': s, 'end': e})

    fine_gradients = depc.get_native('fine_two', fine_rowid_list, 'fine_img')
    coarse_probabilities = depc.get_native(
        'coarse_two', fine_rowid_list, 'coarse_probabilities'
    )
    cropped_images = depc.get_native(
        'preprocess_two', preprocess_rowid_list, 'cropped_img'
    )

    config_ = _convert_depc_config_to_kwargs_config(config)

    contours = ibs.wbia_plugin_curvrank_v2_contours(
        cropped_images, coarse_probabilities, fine_gradients, anchor_points, **config_
    )

    for contour in contours:
        yield (contour,)


class CurvaturesConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('curvrank_width_fine', DEFAULT_WIDTH_FINE['fluke']),
            ut.ParamInfo('curvrank_height_fine', DEFAULT_HEIGHT_FINE['fluke']),
            ut.ParamInfo('curvature_scales', DEFAULT_SCALES['fluke']),
            ut.ParamInfo('curvature_transpose_dims', DEFAULT_TRANSPOSE_DIMS['fluke']),
        ]


@register_preproc_annot(
    tablename='curvature_two',
    parents=['contour_two'],
    colnames=['curvature'],
    coltypes=[np.ndarray],
    configclass=CurvaturesConfig,
    fname='curvrank_v2_unoptimized',
    rm_extern_on_delete=True,
    chunksize=256,
)
# chunksize defines the max number of 'yield' below that will be called in a chunk
# so you would decrease chunksize on expensive calculations
def wbia_plugin_curvrank_v2_curvatures_depc(
    depc,
    contour_rowid_list,
    config=None,
):
    r"""
    Curvatures for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_curvatures_depc
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_curvatures_depc:0
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_curvatures_depc:1
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_curvatures_depc:2

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> curvatures = ibs.depc_annot.get('curvature_two', aid_list, 'curvature', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> curvature = curvatures[0]
        >>> assert ut.hash_data(curvature) in ['byjahrxbzgfkoatkpikcsejvoltxzqid']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> aid_list *= 10
        >>> curvatures = ibs.depc_annot.get('curvature_two', aid_list, 'curvature', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> curvature = curvatures[0]
        >>> assert ut.hash_data(curvature) in ['byjahrxbzgfkoatkpikcsejvoltxzqid']

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(7)
        >>> curvatures = ibs.depc_annot.get('curvature_two', aid_list, 'curvature', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> curvature = curvatures[0]
        >>> assert ut.hash_data(curvature) in ['skyelacotkmcafpytcbqiqyvkswrympm']
    """
    ibs = depc.controller

    contours = depc.get_native('contour_two', contour_rowid_list, 'contour')

    config_ = _convert_depc_config_to_kwargs_config(config)

    curvatures = ibs.wbia_plugin_curvrank_v2_curvatures(contours, **config_)

    for curv in curvatures:
        yield (curv,)


class DescriptorConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('curvrank_scales', DEFAULT_SCALES['fluke']),
            ut.ParamInfo('curvrank_curv_length', 1024),
            ut.ParamInfo('curvrank_feat_dim', 32),
            ut.ParamInfo('curvrank_num_keypoints', 32),
        ]


@register_preproc_annot(
    tablename='curvature_descriptor_two',
    parents=['contour_two', 'curvature_two'],
    colnames=['success', 'descriptor'],
    coltypes=[
        bool,
        (
            'extern',
            ut.partial(ut.load_cPkl, verbose=False),
            ut.partial(ut.save_cPkl, verbose=False),
        ),
    ],
    configclass=DescriptorConfig,
    fname='curvrank_v2_unoptimized',
    rm_extern_on_delete=True,
    chunksize=256,
)
# chunksize defines the max number of 'yield' below that will be called in a chunk
# so you would decrease chunksize on expensive calculations
def wbia_plugin_curvrank_v2_descriptors_depc(
    depc, contour_rowid_list, curvature_rowid_list, config=None
):
    r"""
    Curvature Descriptors for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_descriptors_depc
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_descriptors_depc:0
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_descriptors_depc:1
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_descriptors_depc:2

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> success_list = ibs.depc_annot.get('curvature_descriptor_two', aid_list, 'success', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> descriptors = ibs.depc_annot.get('curvature_descriptor_two', aid_list, 'descriptor', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> curvature_descriptor_dict = descriptors[0]
        >>> hash_list = [
        >>>     ut.hash_data(curvature_descriptor_dict[scale])
        >>>     for scale in sorted(list(curvature_descriptor_dict.keys()))
        >>> ]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(hash_list) in ['ghvpdcfvrvukasxpsoxhzjwyjbbxjzjv']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> aid_list *= 10
        >>> success_list = ibs.depc_annot.get('curvature_descriptor_two', aid_list, 'success', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> descriptors = ibs.depc_annot.get('curvature_descriptor_two', aid_list, 'descriptor', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> success = success_list[0]
        >>> curvature_descriptor_dict = descriptors[0]
        >>> hash_list = [
        >>>     ut.hash_data(curvature_descriptor_dict[scale])
        >>>     for scale in sorted(list(curvature_descriptor_dict.keys()))
        >>> ]
        >>> assert success == True
        >>> assert ut.hash_data(hash_list) in ['ghvpdcfvrvukasxpsoxhzjwyjbbxjzjv']

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(7)
        >>> success_list = ibs.depc_annot.get('curvature_descriptor_two', aid_list, 'success', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> descriptors = ibs.depc_annot.get('curvature_descriptor_two', aid_list, 'descriptor', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> curvature_descriptor_dict = descriptors[0]
        >>> hash_list = [
        >>>     ut.hash_data(curvature_descriptor_dict[scale])
        >>>     for scale in sorted(list(curvature_descriptor_dict.keys()))
        >>> ]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(hash_list) in ['mqxafinoctvyuhljodhqqvsdmfzssuqo']
    """
    ibs = depc.controller

    contours = depc.get_native('contour_two', contour_rowid_list, 'contour')
    curvatures = depc.get_native('curvature_two', curvature_rowid_list, 'curvature')

    config_ = _convert_depc_config_to_kwargs_config(config)

    values = ibs.wbia_plugin_curvrank_v2_descriptors(contours, curvatures, **config_)
    success_list, curvature_descriptor_dicts = values

    for success, curvature_descriptor_dict in zip(
        success_list, curvature_descriptor_dicts
    ):
        yield (
            success,
            curvature_descriptor_dict,
        )


class CurvatureDescriptorOptimizedConfig(dtool.Config):
    def get_param_info_list(self):
        # exclude_key_list = [
        #     'curvrank_daily_cache',
        #     'curvrank_daily_tag',
        #     'curvrank_cache_recompute',
        # ]

        # param_list = []
        # key_list = DEFAULT_DORSAL_TEST_CONFIG.keys()
        # for key in key_list:
        #     if key in exclude_key_list:
        #         continue
        #     value = DEFAULT_DORSAL_TEST_CONFIG[key]
        #     if key.startswith('trailing_edge_finfindr') or key in ['curvrank_greyscale']:
        #         param = ut.ParamInfo(key, value, hideif=value)
        #     else:
        #         param = ut.ParamInfo(key, value)
        #     param_list.append(param)

        param_list = [
            ut.ParamInfo('curvrank_model_type', 'fluke'),
            ut.ParamInfo('curvrank_pad', 0.1),
            ut.ParamInfo('curvrank_width_coarse', DEFAULT_WIDTH_COARSE['fluke']),
            ut.ParamInfo('curvrank_height_coarse', DEFAULT_HEIGHT_COARSE['fluke']),
            ut.ParamInfo('curvrank_width_fine', DEFAULT_WIDTH_FINE['fluke']),
            ut.ParamInfo('curvrank_height_fine', DEFAULT_HEIGHT_FINE['fluke']),
            ut.ParamInfo('curvrank_width_anchor', DEFAULT_WIDTH_ANCHOR['fluke']),
            ut.ParamInfo('curvrank_height_anchor', DEFAULT_HEIGHT_ANCHOR['fluke']),
            ut.ParamInfo('curvrank_patch_size', DEFAULT_PATCH_SIZE['fluke']),
            ut.ParamInfo('curvrank_trim', 0),
            ut.ParamInfo('curvrank_cost_func', 'hyp'),
            ut.ParamInfo('curvrank_scale', DEFAULT_SCALE['fluke']),
            ut.ParamInfo('curvature_scales', DEFAULT_SCALES['fluke']),
            ut.ParamInfo('outline_allow_diagonal', DEFAULT_ALLOW_DIAGONAL['fluke']),
            ut.ParamInfo('curvature_transpose_dims', DEFAULT_TRANSPOSE_DIMS['fluke']),
            ut.ParamInfo('curvature_descriptor_curv_length', 1024),
            ut.ParamInfo('curvature_descriptor_num_keypoints', 32),
            ut.ParamInfo('curvature_descriptor_uniform', False),
            ut.ParamInfo('curvature_descriptor_feat_dim', 32),
            ut.ParamInfo('index_trees', INDEX_NUM_TREES),
            ut.ParamInfo('index_search_k', INDEX_SEARCH_K),
            ut.ParamInfo('index_lnbnn_k', INDEX_LNBNN_K),
        ]

        return param_list


@register_preproc_annot(
    tablename='curvature_descriptor_optimized_two',
    parents=[ROOT],
    colnames=['success', 'descriptor'],
    coltypes=[
        bool,
        (
            'extern',
            ut.partial(ut.load_cPkl, verbose=False),
            ut.partial(ut.save_cPkl, verbose=False),
        ),
    ],
    configclass=CurvatureDescriptorOptimizedConfig,
    fname='curvrank_v2_optimized',
    rm_extern_on_delete=True,
    chunksize=256,
)
# chunksize defines the max number of 'yield' below that will be called in a chunk
# so you would decrease chunksize on expensive calculations
def ibeis_plugin_curvrank_descriptors_optimized_depc(depc, aid_list, config=None):
    r"""
    CommandLine:
        python -m wbia_curvrank_v2._plugin_depc --test-ibeis_plugin_curvrank_descriptors_optimized_depc
        python -m wbia_curvrank_v2._plugin_depc --test-ibeis_plugin_curvrank_descriptors_optimized_depc:0
        python -m wbia_curvrank_v2._plugin_depc --test-ibeis_plugin_curvrank_descriptors_optimized_depc:1
        python -m wbia_curvrank_v2._plugin_depc --test-ibeis_plugin_curvrank_descriptors_optimized_depc:2

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> success_list = ibs.depc_annot.get('curvature_descriptor_optimized_two', aid_list, 'success', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> descriptors = ibs.depc_annot.get('curvature_descriptor_optimized_two', aid_list, 'descriptor', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> curvature_descriptor_dict = descriptors[0]
        >>> hash_list = [
        >>>     ut.hash_data(curvature_descriptor_dict[scale])
        >>>     for scale in sorted(list(curvature_descriptor_dict.keys()))
        >>> ]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(hash_list) in ['ghvpdcfvrvukasxpsoxhzjwyjbbxjzjv']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> success_list = ibs.depc_annot.get('curvature_descriptor_optimized_two', aid_list, 'success', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> descriptors = ibs.depc_annot.get('curvature_descriptor_optimized_two', aid_list, 'descriptor', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> success = success_list[0]
        >>> curvature_descriptor_dict = descriptors[0]
        >>> hash_list = [
        >>>     ut.hash_data(curvature_descriptor_dict[scale])
        >>>     for scale in sorted(list(curvature_descriptor_dict.keys()))
        >>> ]
        >>> assert success == True
        >>> assert ut.hash_data(hash_list) in ['ghvpdcfvrvukasxpsoxhzjwyjbbxjzjv']

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(7)
        >>> success_list = ibs.depc_annot.get('curvature_descriptor_optimized_two', aid_list, 'success', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> descriptors = ibs.depc_annot.get('curvature_descriptor_optimized_two', aid_list, 'descriptor', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> curvature_descriptor_dict = descriptors[0]
        >>> hash_list = [
        >>>     ut.hash_data(curvature_descriptor_dict[scale])
        >>>     for scale in sorted(list(curvature_descriptor_dict.keys()))
        >>> ]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(hash_list) in ['mqxafinoctvyuhljodhqqvsdmfzssuqo']
    """
    ibs = depc.controller

    config_ = _convert_depc_config_to_kwargs_config(config)

    values = ibs.wbia_plugin_curvrank_v2_pipeline_compute(aid_list, config_)
    success_list, curvature_descriptor_dicts = values

    for success, curvature_descriptor_dict in zip(
        success_list, curvature_descriptor_dicts
    ):
        yield (
            success,
            curvature_descriptor_dict,
        )


@register_ibs_method
def wbia_plugin_curvrank_v2_scores_depc(ibs, db_aid_list, qr_aid_list, **kwargs):
    r"""
    CurvRank Example

    Args:
        ibs           (IBEISController): IBEIS controller object
        db_aid_list   (list of ints): database annotation rowids
        qr_aids_list  (list of ints): query annotation rowids

    Returns:
        score_dict

    CommandLine:
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_scores_depc
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_scores_depc:0
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_scores_depc:1
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_scores_depc:2

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> db_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Fluke Database')
        >>> db_aid_list = ibs.get_imageset_aids(db_imageset_rowid)
        >>> qr_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Fluke Query')
        >>> qr_aid_list = ibs.get_imageset_aids(qr_imageset_rowid)
        >>> config = DEFAULT_FLUKE_TEST_CONFIG
        >>> config['curvrank_daily_cache'] = False
        >>> score_dict_iter = ibs.wbia_plugin_curvrank_v2_scores_depc(db_aid_list, [qr_aid_list], config=config)
        >>> score_dict_list = list(score_dict_iter)
        >>> qr_aid_list, score_dict = score_dict_list[0]
        >>> for key in score_dict:
        >>>     score_dict[key] = round(score_dict[key], 2)
        >>> result = sorted(score_dict.items())
        >>> print(result)
        [(7, -0.64), (8, -0.45), (9, -0.24), (10, -0.27), (11, -7.43), (12, -0.49), (13, -0.81), (14, -0.72)]

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> db_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Fluke Database')
        >>> db_aid_list = ibs.get_imageset_aids(db_imageset_rowid)
        >>> qr_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Fluke Query')
        >>> qr_aid_list = ibs.get_imageset_aids(qr_imageset_rowid)
        >>> config = DEFAULT_FLUKE_TEST_CONFIG
        >>> config['curvrank_daily_cache'] = False
        >>> score_dict_iter = ibs.wbia_plugin_curvrank_v2_scores_depc(db_aid_list, [qr_aid_list], config=config, use_depc_optimized=True)
        >>> score_dict_list = list(score_dict_iter)
        >>> qr_aid_list, score_dict = score_dict_list[0]
        >>> for key in score_dict:
        >>>     score_dict[key] = round(score_dict[key], 2)
        >>> result = sorted(score_dict.items())
        >>> print(result)
        [(7, -0.64), (8, -0.45), (9, -0.24), (10, -0.27), (11, -7.43), (12, -0.49), (13, -0.81), (14, -0.72)]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> db_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Dorsal Database')
        >>> db_aid_list = ibs.get_imageset_aids(db_imageset_rowid)
        >>> qr_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Dorsal Query')
        >>> qr_aid_list = ibs.get_imageset_aids(qr_imageset_rowid)
        >>> config = DEFAULT_DORSAL_TEST_CONFIG
        >>> config['curvrank_daily_cache'] = False
        >>> score_dict_iter = ibs.wbia_plugin_curvrank_v2_scores_depc(db_aid_list, [qr_aid_list], config=config, use_depc_optimized=True)
        >>> score_dict_list = list(score_dict_iter)
        >>> qr_aid_list, score_dict = score_dict_list[0]
        >>> for key in score_dict:
        >>>     score_dict[key] = round(score_dict[key], 2)
        >>> result = sorted(score_dict.items())
        >>> print(result)
        [(1, -22.31), (2, -1.74), (3, -2.49)]
    """
    kwargs['use_depc'] = True
    kwargs['config'] = _convert_depc_config_to_kwargs_config(kwargs.get('config', {}))
    return ibs.wbia_plugin_curvrank_v2_scores(db_aid_list, qr_aid_list, **kwargs)


def get_match_results(depc, qaid_list, daid_list, score_list, config):
    """ converts table results into format for ipython notebook """
    # qaid_list, daid_list = request.get_parent_rowids()
    # score_list = request.score_list
    # config = request.config

    unique_qaids, groupxs = ut.group_indices(qaid_list)
    # grouped_qaids_list = ut.apply_grouping(qaid_list, groupxs)
    grouped_daids = ut.apply_grouping(daid_list, groupxs)
    grouped_scores = ut.apply_grouping(score_list, groupxs)

    ibs = depc.controller
    unique_qnids = ibs.get_annot_nids(unique_qaids)

    # scores
    _iter = zip(unique_qaids, unique_qnids, grouped_daids, grouped_scores)
    for qaid, qnid, daids, scores in _iter:
        dnids = ibs.get_annot_nids(daids)

        # Remove distance to self
        annot_scores = np.array(scores)
        daid_list_ = np.array(daids)
        dnid_list_ = np.array(dnids)

        is_valid = daid_list_ != qaid
        daid_list_ = daid_list_.compress(is_valid)
        dnid_list_ = dnid_list_.compress(is_valid)
        annot_scores = annot_scores.compress(is_valid)

        # Hacked in version of creating an annot match object
        match_result = wbia.AnnotMatch()
        match_result.qaid = qaid
        match_result.qnid = qnid
        match_result.daid_list = daid_list_
        match_result.dnid_list = dnid_list_
        match_result._update_daid_index()
        match_result._update_unique_nid_index()

        grouped_annot_scores = vt.apply_grouping(annot_scores, match_result.name_groupxs)
        name_scores = np.array([np.sum(dists) for dists in grouped_annot_scores])
        match_result.set_cannonical_name_score(annot_scores, name_scores)
        yield match_result


class CurvRankRequest(dtool.base.VsOneSimilarityRequest):  # NOQA
    _symmetric = False

    def overlay_trailing_edge(
        request, chip, outline, trailing_edge, edge_color=(0, 255, 255)
    ):
        chip_ = np.copy(chip)
        ratio = request.config.curvrank_width_fine / chip_.shape[1]
        chip_ = cv2.resize(
            chip_, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA
        )
        h, w = chip_.shape[:2]

        if outline is not None:
            for y, x in outline:
                if x < 0 or w < x or y < 0 or h < y:
                    continue
                cv2.circle(chip_, (x, y), 5, (255, 0, 0), thickness=-1)

        if trailing_edge is not None:
            for y, x in trailing_edge:
                if x < 0 or w < x or y < 0 or h < y:
                    continue
                cv2.circle(chip_, (x, y), 2, edge_color, thickness=-1)

        return chip_

    @ut.accepts_scalar_input
    def get_fmatch_overlayed_chip(request, aid_list, overlay=True, config=None):
        depc = request.depc

        chips = depc.get('preprocess_two', aid_list, 'cropped_img', config=request.config)
        outlines = [None] * len(chips)
        trailing_edges = [None] * len(chips)

        if overlay:
            trailing_edges = depc.get(
                'contour_two', aid_list, 'contour', config=request.config
            )

        overlay_chips = [
            request.overlay_trailing_edge(chip, outline, trailing_edge)
            for chip, outline, trailing_edge in zip(chips, outlines, trailing_edges)
        ]
        return overlay_chips

    def render_single_result(request, cm, aid, **kwargs):
        # HACK FOR WEB VIEWER
        overlay = kwargs.get('draw_fmatches')
        chips = request.get_fmatch_overlayed_chip(
            [cm.qaid, aid], overlay=overlay, config=request.config
        )
        import vtool as vt

        out_img = vt.stack_image_list(chips)
        return out_img

    def postprocess_execute(request, parent_rowids, result_list):
        qaid_list, daid_list = list(zip(*parent_rowids))
        score_list = ut.take_column(result_list, 0)
        depc = request.depc
        config = request.config
        cm_list = list(get_match_results(depc, qaid_list, daid_list, score_list, config))
        return cm_list

    def execute(request, *args, **kwargs):
        kwargs['use_cache'] = False
        result_list = super(CurvRankRequest, request).execute(*args, **kwargs)
        qaids = kwargs.pop('qaids', None)
        if qaids is not None:
            result_list = [result for result in result_list if result.qaid in qaids]
        return result_list


class CurvRankDorsalConfig(dtool.Config):  # NOQA
    """
    CommandLine:
        python -m wbia_curvrank_v2._plugin_depc --test-CurvRankDorsalConfig

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> config = CurvRankDorsalConfig()
        >>> result = config.get_cfgstr()
        >>> print(result)
        CurvRankDorsal(curvature_descriptor_curv_length=1024,curvature_descriptor_feat_dim=32,curvature_descriptor_num_keypoints=32,curvature_descriptor_uniform=False,curvature_scales=[0.04 0.06 0.08 0.1 ],curvature_transpose_dims=False,curvrank_cache_recompute=False,curvrank_cost_func=hyp,curvrank_daily_cache=True,curvrank_daily_tag=global,curvrank_height_anchor=224,curvrank_height_coarse=256,curvrank_height_fine=1024,curvrank_model_type=dorsal,curvrank_pad=0.1,curvrank_patch_size=None,curvrank_scale=4,curvrank_trim=0,curvrank_width_anchor=224,curvrank_width_coarse=256,curvrank_width_fine=1024,outline_allow_diagonal=False)
    """

    def get_param_info_list(self):
        param_list = []
        key_list = DEFAULT_DORSAL_TEST_CONFIG.keys()
        for key in sorted(key_list):
            value = DEFAULT_DORSAL_TEST_CONFIG[key]
            if key.startswith('trailing_edge_finfindr_') or key.startswith('index_'):
                param = ut.ParamInfo(key, value, hideif=value)
            else:
                param = ut.ParamInfo(key, value)
            param_list.append(param)
        return param_list


class CurvRankDorsalRequest(CurvRankRequest):  # NOQA
    _tablename = 'CurvRankTwoDorsal'


@register_preproc_annot(
    tablename='CurvRankTwoDorsal',
    parents=[ROOT, ROOT],
    colnames=['score'],
    coltypes=[float],
    configclass=CurvRankDorsalConfig,
    requestclass=CurvRankDorsalRequest,
    fname='curvrank_v2_scores_dorsal',
    rm_extern_on_delete=True,
    chunksize=None,
)
def wbia_plugin_curvrank_v2_dorsal(depc, qaid_list, daid_list, config):
    r"""
    CommandLine:
        python -m wbia_curvrank_v2._plugin_depc --exec-wbia_plugin_curvrank_v2_dorsal --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> import itertools as it
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> depc = ibs.depc_annot
        >>> imageset_rowid_list = ibs.get_imageset_imgsetids_from_text(['Dorsal Database', 'Dorsal Query'])
        >>> aid_list = list(set(ut.flatten(ibs.get_imageset_aids(imageset_rowid_list))))
        >>> root_rowids = tuple(zip(*it.product(aid_list, aid_list)))
        >>> qaid_list, daid_list = root_rowids
        >>> config = CurvRankDorsalConfig()
        >>> # Call function via request
        >>> request = CurvRankDorsalRequest.new(depc, aid_list, aid_list)
        >>> am_list1 = request.execute()
        >>> # Call function via depcache
        >>> prop_list = depc.get('CurvRankTwoDorsal', root_rowids)
        >>> # Call function normally
        >>> score_list = list(wbia_plugin_curvrank_v2_dorsal(depc, qaid_list, daid_list, config))
        >>> am_list2 = list(get_match_results(depc, qaid_list, daid_list, score_list, config))
        >>> assert score_list == prop_list, 'error in cache'
        >>> assert np.all(am_list1[0].score_list == am_list2[0].score_list)
        >>> ut.quit_if_noshow()
        >>> am = am_list2[0]
        >>> am.ishow_analysis(request)
        >>> ut.show_if_requested()
    """

    ibs = depc.controller

    label = 'CurvRankTwoDorsal'
    value_iter = ibs.wbia_plugin_curvrank_v2(label, qaid_list, daid_list, config)
    for value in value_iter:
        yield value


class CurvRankFlukeConfig(dtool.Config):  # NOQA
    """
    CommandLine:
        python -m wbia_curvrank_v2._plugin_depc --test-CurvRankFlukeConfig

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> config = CurvRankFlukeConfig()
        >>> result = config.get_cfgstr()
        >>> print(result)
        CurvRankFluke(curvature_descriptor_curv_length=1024,curvature_descriptor_feat_dim=32,curvature_descriptor_num_keypoints=32,curvature_descriptor_uniform=False,curvature_scales=[0.02 0.04 0.06 0.08],curvature_transpose_dims=True,curvrank_cache_recompute=False,curvrank_cost_func=exp,curvrank_daily_cache=True,curvrank_daily_tag=global,curvrank_height_anchor=224,curvrank_height_coarse=192,curvrank_height_fine=576,curvrank_model_type=fluke,curvrank_pad=0.1,curvrank_patch_size=128,curvrank_scale=3,curvrank_trim=0,curvrank_width_anchor=224,curvrank_width_coarse=384,curvrank_width_fine=1152,outline_allow_diagonal=True)
    """

    def get_param_info_list(self):
        param_list = []
        key_list = DEFAULT_FLUKE_TEST_CONFIG.keys()
        for key in sorted(key_list):
            value = DEFAULT_FLUKE_TEST_CONFIG[key]
            if key.startswith('trailing_edge_finfindr_') or key.startswith('index_'):
                param = ut.ParamInfo(key, value, hideif=value)
            else:
                param = ut.ParamInfo(key, value)
            param_list.append(param)
        return param_list


class CurvRankFlukeRequest(CurvRankRequest):  # NOQA
    _tablename = 'CurvRankTwoFluke'


@register_preproc_annot(
    tablename='CurvRankTwoFluke',
    parents=[ROOT, ROOT],
    colnames=['score'],
    coltypes=[float],
    configclass=CurvRankFlukeConfig,
    requestclass=CurvRankFlukeRequest,
    fname='curvrank_v2_scores_fluke',
    rm_extern_on_delete=True,
    chunksize=None,
)
def wbia_plugin_curvrank_v2_fluke(depc, qaid_list, daid_list, config):
    r"""
    CommandLine:
        python -m wbia_curvrank_v2._plugin_depc --test-wbia_plugin_curvrank_v2_fluke --show

    Example0:
        >>> # DISABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> import itertools as it
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> depc = ibs.depc_annot
        >>> imageset_rowid_list = ibs.get_imageset_imgsetids_from_text(['Fluke Database', 'Fluke Query'])
        >>> aid_list = list(set(ut.flatten(ibs.get_imageset_aids(imageset_rowid_list))))
        >>> root_rowids = tuple(zip(*it.product(aid_list, aid_list)))
        >>> qaid_list, daid_list = root_rowids
        >>> config = CurvRankFlukeConfig()
        >>> # Call function via request
        >>> request = CurvRankFlukeRequest.new(depc, aid_list, aid_list)
        >>> am_list1 = request.execute()
        >>> # Call function via depcache
        >>> prop_list = depc.get('CurvRankTwoFluke', root_rowids)
        >>> # Call function normally
        >>> score_list = list(wbia_plugin_curvrank_v2_fluke(depc, qaid_list, daid_list, config))
        >>> am_list2 = list(get_match_results(depc, qaid_list, daid_list, score_list, config))
        >>> assert score_list == prop_list, 'error in cache'
        >>> assert np.all(am_list1[0].score_list == am_list2[0].score_list)
        >>> ut.quit_if_noshow()
        >>> am_list2 = [am for am in am_list2 if am.qaid == 23]
        >>> am = am_list2[0]
        >>> am.ishow_analysis(request)
        >>> ut.show_if_requested()

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin_depc import *  # NOQA
        >>> import wbia
        >>> import itertools as it
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> depc = ibs.depc_annot
        >>> imageset_rowid_list = ibs.get_imageset_imgsetids_from_text(['Fluke Database', 'Fluke Query'])
        >>> aid_list = list(set(ut.flatten(ibs.get_imageset_aids(imageset_rowid_list))))
        >>> root_rowids = tuple(zip(*it.product(aid_list, aid_list)))
        >>> qaid_list, daid_list = root_rowids
        >>> config = CurvRankFlukeConfig()
        >>> # Call function via request
        >>> request = CurvRankFlukeRequest.new(depc, aid_list, aid_list)
        >>> am_list1 = request.execute()
        >>> ut.quit_if_noshow()
        >>> am_list2 = [am for am in am_list1 if am.qaid == 23]
        >>> am = am_list2[0]
        >>> am.ishow_analysis(request)
        >>> ut.show_if_requested()
    """
    ibs = depc.controller

    label = 'CurvRankTwoFluke'
    value_iter = ibs.wbia_plugin_curvrank_v2(label, qaid_list, daid_list, config)
    for value in value_iter:
        yield value


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia_curvrank_v2._plugin_depc --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
