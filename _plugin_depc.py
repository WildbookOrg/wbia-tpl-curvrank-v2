from __future__ import absolute_import, division, print_function
from ibeis.control import controller_inject  # NOQA
import numpy as np
import utool as ut
import vtool as vt
import dtool
import ibeis


_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)

register_preproc_annot = controller_inject.register_preprocs['annot']


DEFAULT_WIDTH  = {
    'dorsal': 256,
    'fluke' : 384,
}
DEFAULT_HEIGHT = {
    'dorsal': 256,
    'fluke' : 192,
}
DEFAULT_SCALE  = {
    'dorsal': 4,
    'fluke' : 3,
}
DEFAULT_SCALES = {
    'dorsal': np.array([0.04, 0.06, 0.08, 0.10], dtype=np.float32),
    'fluke' : np.array([0.02, 0.04, 0.06, 0.08], dtype=np.float32),
}
DEFAULT_ALLOW_DIAGONAL = {
    'dorsal': False,
    'fluke':  True,
}
DEFAULT_TRANSPOSE_DIMS = {
    'dorsal': False,
    'fluke':  True,
}


INDEX_NUM_TREES  = 10
INDEX_NUM_ANNOTS = 2000  # 1000
INDEX_LNBNN_K    = 2
INDEX_SEARCH_D   = 2  # 1
INDEX_SEARCH_K   = INDEX_LNBNN_K * INDEX_NUM_TREES * INDEX_SEARCH_D


DEFAULT_DORSAL_TEST_CONFIG = {
    'curvrank_daily_cache'                 : True,
    'curvrank_daily_tag'                   : 'global',
    'curvrank_cache_recompute'             : False,
    'curvrank_model_type'                  : 'dorsal',
    'curvrank_width'                       : DEFAULT_WIDTH['dorsal'],
    'curvrank_height'                      : DEFAULT_HEIGHT['dorsal'],
    'curvrank_greyscale'                   : False,
    'curvrank_scale'                       : DEFAULT_SCALE['dorsal'],
    'curvature_scales'                     : DEFAULT_SCALES['dorsal'],
    'outline_allow_diagonal'               : DEFAULT_ALLOW_DIAGONAL['dorsal'],
    'curvatute_transpose_dims'             : DEFAULT_TRANSPOSE_DIMS['dorsal'],
    'localization_model_tag'               : 'localization',
    'segmentation_model_tag'               : 'segmentation',
    'segmentation_gt_radius'               : 25,
    'segmentation_gt_opacity'              : 0.5,
    'segmentation_gt_smooth'               : True,
    'segmentation_gt_smooth_margin'        : 0.001,
    'trailing_edge_finfindr_smooth'        : True,
    'trailing_edge_finfindr_smooth_margin' : 0.0,
    'curvature_descriptor_curv_length'     : 1024,
    'curvature_descriptor_num_keypoints'   : 32,
    'curvature_descriptor_uniform'         : False,
    'curvature_descriptor_feat_dim'        : 32,
    'index_trees'                          : INDEX_NUM_TREES,
    'index_search_k'                       : INDEX_SEARCH_K,
    'index_lnbnn_k'                        : INDEX_LNBNN_K,
}


DEFAULT_FLUKE_TEST_CONFIG = {
    'curvrank_daily_cache'                 : True,
    'curvrank_daily_tag'                   : 'global',
    'curvrank_cache_recompute'             : False,
    'curvrank_model_type'                  : 'fluke',
    'curvrank_width'                       : DEFAULT_WIDTH['fluke'],
    'curvrank_height'                      : DEFAULT_HEIGHT['fluke'],
    'curvrank_greyscale'                   : True,
    'curvrank_scale'                       : DEFAULT_SCALE['fluke'],
    'curvature_scales'                     : DEFAULT_SCALES['fluke'],
    'outline_allow_diagonal'               : DEFAULT_ALLOW_DIAGONAL['fluke'],
    'curvatute_transpose_dims'             : DEFAULT_TRANSPOSE_DIMS['fluke'],
    'localization_model_tag'               : 'localization',
    'segmentation_model_tag'               : 'segmentation',
    'segmentation_gt_radius'               : 25,
    'segmentation_gt_opacity'              : 0.5,
    'segmentation_gt_smooth'               : True,
    'segmentation_gt_smooth_margin'        : 0.001,
    'trailing_edge_finfindr_smooth'        : True,
    'trailing_edge_finfindr_smooth_margin' : 0.0,
    'curvature_descriptor_curv_length'     : 1024,
    'curvature_descriptor_num_keypoints'   : 32,
    'curvature_descriptor_uniform'         : False,
    'curvature_descriptor_feat_dim'        : 32,
    'index_trees'                          : INDEX_NUM_TREES,
    'index_search_k'                       : INDEX_SEARCH_K,
    'index_lnbnn_k'                        : INDEX_LNBNN_K,
}


DEFAULT_DEPC_KEY_MAPPING = {
    'curvrank_daily_cache'                 : 'use_daily_cache',
    'curvrank_daily_tag'                   : 'daily_cache_tag',
    'curvrank_cache_recompute'             : 'force_cache_recompute',
    'curvrank_model_type'                  : 'model_type',
    'curvrank_width'                       : 'width',
    'curvrank_height'                      : 'height',
    'curvrank_greyscale'                   : 'greyscale',
    'curvrank_scale'                       : 'scale',
    'curvature_scales'                     : 'scales',
    'outline_allow_diagonal'               : 'allow_diagonal',
    'curvatute_transpose_dims'             : 'transpose_dims',
    'segmentation_gt_radius'               : 'groundtruth_radius',
    'segmentation_gt_opacity'              : 'groundtruth_opacity',
    'segmentation_gt_smooth'               : 'groundtruth_smooth',
    'segmentation_gt_smooth_margin'        : 'groundtruth_smooth_margin',
    'trailing_edge_finfindr_smooth'        : 'finfindr_smooth',
    'trailing_edge_finfindr_smooth_margin' : 'finfindr_smooth_margin',
    'curvature_descriptor_curv_length'     : 'curv_length',
    'curvature_descriptor_num_keypoints'   : 'num_keypoints',
    'curvature_descriptor_uniform'         : 'uniform',
    'curvature_descriptor_feat_dim'        : 'feat_dim',
    'index_trees'                          : 'num_trees',
    'index_search_k'                       : 'search_k',
    'index_lnbnn_k'                        : 'lnbnn_k',
}


ROOT = ibeis.const.ANNOTATION_TABLE


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
            ut.ParamInfo('curvrank_height',     DEFAULT_HEIGHT['dorsal']),
            ut.ParamInfo('curvrank_width',      DEFAULT_WIDTH['dorsal']),
            ut.ParamInfo('curvrank_greyscale',  False, hideif=False),
            ut.ParamInfo('ext',                 '.npy', hideif='.npy'),
        ]


@register_preproc_annot(
    tablename='preprocess', parents=[ROOT],
    colnames=['resized_img', 'resized_width', 'resized_height', 'mask_img', 'mask_width', 'mask_height', 'pretransform'],
    coltypes=[('extern', np.load, np.save), int, int, ('extern', np.load, np.save), int, int, np.ndarray],
    configclass=PreprocessConfig,
    fname='curvrank_unoptimized',
    rm_extern_on_delete=True,
    chunksize=256,
)
def ibeis_plugin_curvrank_preprocessing_depc(depc, aid_list, config=None):
    r"""
    Pre-process images for CurvRank with Dependency Cache (depc)

    Args:
        depc      (Dependency Cache): IBEIS dependency cache object
        aid_list  (list of int): list of annot rowids (aids)
        config    (PreprocessConfig): config for depcache

    CommandLine:
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_preprocessing_depc
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_preprocessing_depc:0
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_preprocessing_depc:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> resized_images = ibs.depc_annot.get('preprocess', aid_list, 'resized_img',  config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> resized_masks  = ibs.depc_annot.get('preprocess', aid_list, 'mask_img',     config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> pre_transforms = ibs.depc_annot.get('preprocess', aid_list, 'pretransform', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> resized_image = resized_images[0]
        >>> resized_mask  = resized_masks[0]
        >>> pre_transform = pre_transforms[0]
        >>> assert ut.hash_data(resized_image) in ['inxtvdeyxibloygwuyhxzpnevpkoenec']
        >>> assert ut.hash_data(resized_mask)  in ['mnhartnytowmmhskblocubqmzhbofynr']
        >>> result = pre_transform
        >>> print(result)
        [[ 0.36571429  0.          0.        ]
         [ 0.          0.36571429 38.        ]
         [ 0.          0.          1.        ]]

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> resized_images = ibs.depc_annot.get('preprocess', aid_list, 'resized_img',  config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> resized_masks  = ibs.depc_annot.get('preprocess', aid_list, 'mask_img',     config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> pre_transforms = ibs.depc_annot.get('preprocess', aid_list, 'pretransform', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> resized_image = resized_images[0]
        >>> resized_mask  = resized_masks[0]
        >>> pre_transform = pre_transforms[0]
        >>> assert ut.hash_data(resized_image) in ['rygggbfijzssfanhlvbchlyxdvaltuvy']
        >>> assert ut.hash_data(resized_mask)  in ['xrecwbobdxovkrzojngixulmmegimxwv']
        >>> result = pre_transform
        >>> print(result)
        [[ 0.54857143  0.          0.        ]
         [ 0.          0.54857143 52.        ]
         [ 0.          0.          1.        ]]
    """
    ibs = depc.controller

    width     = config['curvrank_width']
    height    = config['curvrank_height']
    greyscale = config['curvrank_greyscale']

    values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list, width=width, height=height,
                                                     greyscale=greyscale)
    resized_images, resized_masks, pre_transforms = values

    zipped = zip(resized_images, resized_masks, pre_transforms)
    for resized_image, resized_mask, pre_transform in zipped:
        resized_width, resized_height = vt.get_size(resized_image)
        mask_width, mask_height = vt.get_size(resized_mask)

        yield (
            resized_image,
            resized_width,
            resized_height,
            resized_mask,
            mask_width,
            mask_height,
            pre_transform,
        )


class LocalizationConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('curvrank_model_type',     'dorsal'),
            ut.ParamInfo('curvrank_height',         DEFAULT_HEIGHT['dorsal']),
            ut.ParamInfo('curvrank_width',          DEFAULT_WIDTH['dorsal']),
            ut.ParamInfo('localization_model_tag',  'localization'),
            ut.ParamInfo('ext',                     '.npy', hideif='.npy'),
        ]


@register_preproc_annot(
    tablename='localization', parents=['preprocess'],
    colnames=['localized_img', 'localized_width', 'localized_height', 'mask_img', 'mask_width', 'mask_height', 'transform'],
    coltypes=[('extern', np.load, np.save), int, int, ('extern', np.load, np.save), int, int, np.ndarray],
    configclass=LocalizationConfig,
    fname='curvrank_unoptimized',
    rm_extern_on_delete=True,
    chunksize=128,
)
# chunksize defines the max number of 'yield' below that will be called in a chunk
# so you would decrease chunksize on expensive calculations
def ibeis_plugin_curvrank_localization_depc(depc, preprocess_rowid_list, config=None):
    r"""
    Localize images for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_localization_depc
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_localization_depc:0
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_localization_depc:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> localized_images = ibs.depc_annot.get('localization', aid_list, 'localized_img',  config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> localized_masks  = ibs.depc_annot.get('localization', aid_list, 'mask_img',     config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> loc_transforms = ibs.depc_annot.get('localization', aid_list, 'transform', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> localized_image = localized_images[0]
        >>> localized_mask  = localized_masks[0]
        >>> loc_transform = loc_transforms[0]
        >>> # localized_image appears to differ very slightly in ubuntu vs. mac. Hashes below for each respectively.
        >>> #TODO verify that mac/ubuntu values are consistent on those OSes
        >>> assert ut.hash_data(localized_image) in ['igxwfzwvpbqpfriihmdsyaoksqbzviey']
        >>> assert ut.hash_data(localized_mask)  in ['whrbbdtqbmpyjskvvpvblehfiofdgsli']
        >>> # for above reasons, some voodoo to compare loc_transform
        >>> loc_transform_ubuntu = np.array([[ 6.42954651e-01,  1.20030158e-01, -1.06427952e-01],
                                             [-1.19038359e-01,  6.43158788e-01, -1.27811638e-04],
                                             [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        >>> assert np.all(np.abs(loc_transform - loc_transform_ubuntu) < 1e-6)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> resized_images = ibs.depc_annot.get('preprocess', aid_list, 'resized_img',  config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> resized_masks  = ibs.depc_annot.get('preprocess', aid_list, 'mask_img',     config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> pre_transforms = ibs.depc_annot.get('preprocess', aid_list, 'pretransform', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> localized_images = ibs.depc_annot.get('localization', aid_list, 'localized_img', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> localized_masks  = ibs.depc_annot.get('localization', aid_list, 'mask_img',      config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> loc_transforms   = ibs.depc_annot.get('localization', aid_list, 'transform',     config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> localized_image = localized_images[0]
        >>> localized_mask  = localized_masks[0]
        >>> loc_transform = loc_transforms[0]
        >>> # localized_image appears to differ very slightly in ubuntu vs. mac. Hashes below for each respectively.
        >>> #TODO verify that mac/ubuntu values are consistent on those OSes
        >>> assert ut.hash_data(resized_images) == ut.hash_data(localized_images)
        >>> assert ut.hash_data(resized_masks)  == ut.hash_data(localized_masks)
        >>> assert np.sum(loc_transform) == 3.0
    """
    ibs = depc.controller

    model_type = config['curvrank_model_type']
    width      = config['curvrank_width']
    height     = config['curvrank_height']
    model_tag  = config['localization_model_tag']

    resized_images = depc.get_native('preprocess', preprocess_rowid_list, 'resized_img')
    resized_masks  = depc.get_native('preprocess', preprocess_rowid_list, 'mask_img')

    values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks,
                                                    width=width, height=height,
                                                    model_type=model_type,
                                                    model_tag=model_tag)
    localized_images, localized_masks, loc_transforms = values

    # yield each column defined in register_preproc_annot
    zipped = zip(localized_images, localized_masks, loc_transforms)
    for localized_image, localized_mask, loc_transform in zipped:
        localized_width, localized_height = vt.get_size(localized_image)
        mask_width, mask_height = vt.get_size(localized_mask)
        yield (
            localized_image,
            localized_width,
            localized_height,
            localized_mask,
            mask_width,
            mask_height,
            loc_transform,
        )


class RefinementConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('curvrank_width',      DEFAULT_HEIGHT['dorsal']),
            ut.ParamInfo('curvrank_height',     DEFAULT_WIDTH['dorsal']),
            ut.ParamInfo('curvrank_scale',      DEFAULT_SCALE['dorsal']),
            ut.ParamInfo('curvrank_greyscale',  False, hideif=False),
            ut.ParamInfo('ext',                 '.npy', hideif='.npy'),
        ]


@register_preproc_annot(
    tablename='refinement', parents=['localization', 'preprocess'],
    colnames=['refined_img', 'refined_width', 'refined_height', 'mask_img', 'mask_width', 'mask_height'],
    coltypes=[('extern', np.load, np.save), int, int, ('extern', np.load, np.save), int, int],
    configclass=RefinementConfig,
    fname='curvrank_unoptimized',
    rm_extern_on_delete=True,
    chunksize=256,
)
# chunksize defines the max number of 'yield' below that will be called in a chunk
# so you would decrease chunksize on expensive calculations
def ibeis_plugin_curvrank_refinement_depc(depc, localization_rowid_list,
                                          preprocess_rowid_list, config=None):
    r"""
    Refine localizations for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_refinement_depc
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_refinement_depc:0
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_refinement_depc:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> refined_localizations = ibs.depc_annot.get('refinement', aid_list, 'refined_img', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> refined_masks         = ibs.depc_annot.get('refinement', aid_list, 'mask_img', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> refined_localization  = refined_localizations[0]
        >>> refined_mask          = refined_masks[0]
        >>> #TODO verify that mac/ubuntu values are consistent on those OSes
        >>> # why are these values different than in above? have we cached bad stuff? I'm guessing yes.
        >>> assert ut.hash_data(refined_localization) in ['nxhumkmybgbjdjcffuneozzmptvivvlh']
        >>> assert ut.hash_data(refined_mask)         in ['bwuzcdgbfyqhzgdthazfgegbzeykvbnt']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> refined_localizations = ibs.depc_annot.get('refinement', aid_list, 'refined_img', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> refined_masks         = ibs.depc_annot.get('refinement', aid_list, 'mask_img', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> refined_localization  = refined_localizations[0]
        >>> refined_mask          = refined_masks[0]
        >>> #TODO verify that mac/ubuntu values are consistent on those OSes
        >>> # why are these values different than in above? have we cached bad stuff? I'm guessing yes.
        >>> assert ut.hash_data(refined_localization) in ['cwmqsvpabxdaftsnupgerivjufsavfhl']
        >>> assert ut.hash_data(refined_mask)         in ['zwfgmumqblkfzejnseauggiedzpbbjoa']
    """
    ibs = depc.controller

    width     = config['curvrank_width']
    height    = config['curvrank_height']
    scale     = config['curvrank_scale']
    greyscale = config['curvrank_greyscale']

    aid_list = depc.get_ancestor_rowids('preprocess',  preprocess_rowid_list)
    loc_transforms   = depc.get_native('localization', localization_rowid_list, 'transform')
    pre_transforms   = depc.get_native('preprocess',   preprocess_rowid_list,   'pretransform')

    values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms,
                                                  width=width, height=height, scale=scale,
                                                  greyscale=greyscale)
    refined_localizations, refined_masks = values

    for refined_localization, refined_mask in zip(refined_localizations, refined_masks):
        refined_localization_height, refined_localization_width = refined_localization.shape[:2]
        refined_mask_height, refined_mask_width = refined_mask.shape[:2]
        yield (
            refined_localization,
            refined_localization_width,
            refined_localization_height,
            refined_mask,
            refined_mask_width,
            refined_mask_height
        )


class SegmentationConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('curvrank_model_type',           'dorsal'),
            ut.ParamInfo('curvrank_width',                DEFAULT_HEIGHT['dorsal']),
            ut.ParamInfo('curvrank_height',               DEFAULT_WIDTH['dorsal']),
            ut.ParamInfo('curvrank_scale',                DEFAULT_SCALE['dorsal']),
            ut.ParamInfo('segmentation_model_tag',        'segmentation'),
            ut.ParamInfo('segmentation_gt_radius',        25),
            ut.ParamInfo('segmentation_gt_opacity',       0.5),
            ut.ParamInfo('segmentation_gt_smooth',        True),
            ut.ParamInfo('segmentation_gt_smooth_margin', 0.001),
            ut.ParamInfo('curvrank_greyscale',            False, hideif=False),
            ut.ParamInfo('ext',                           '.npy', hideif='.npy'),
        ]


@register_preproc_annot(
    tablename='segmentation', parents=['refinement', 'preprocess', 'localization'],
    colnames=['segmentations_img', 'refined_width', 'refined_height', 'refined_segmentations_img', 'refined_segmentations_width', 'refined_segmentations_height'],
    coltypes=[('extern', np.load, np.save), int, int, ('extern', np.load, np.save), int, int],
    configclass=SegmentationConfig,
    fname='curvrank_unoptimized',
    rm_extern_on_delete=True,
    chunksize=128,
)
# chunksize defines the max number of 'yield' below that will be called in a chunk
# so you would decrease chunksize on expensive calculations
def ibeis_plugin_curvrank_segmentation_depc(depc, refinement_rowid_list, preprocess_rowid_list,
                                            localization_rowid_list, config=None):
    r"""
    Refine localizations for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_segmentation_depc
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_segmentation_depc:0
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_segmentation_depc:1
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_segmentation_depc:2
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_segmentation_depc:3

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> segmentations          = ibs.depc_annot.get('segmentation', aid_list, 'segmentations_img', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> refined_segmentations  = ibs.depc_annot.get('segmentation', aid_list, 'refined_segmentations_img', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> segmentation           = segmentations[0]
        >>> refined_segmentation   = refined_segmentations[0]
        >>> assert ut.hash_data(segmentation)         in ['tcfybjuqszadvmfetzxivcvihfkudvqh']
        >>> assert ut.hash_data(refined_segmentation) in ['snjswkyqprmhmpefiiiapdsytubfvcwo']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> segmentations          = ibs.depc_annot.get('segmentation', aid_list, 'segmentations_img', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> refined_segmentations  = ibs.depc_annot.get('segmentation', aid_list, 'refined_segmentations_img', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> segmentation           = segmentations[0]
        >>> refined_segmentation   = refined_segmentations[0]
        >>> assert ut.hash_data(segmentation)         in ['htbsspdnjfchswtcboifeybpkhmbdmms']
        >>> assert ut.hash_data(refined_segmentation) in ['hqngsbdctbjsruuwjhhbuamcbukbyaea']

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list, part_rowid_list = ibs.ibeis_plugin_curvrank_test_setup_groundtruth()
        >>> try:
        >>>     config = DEFAULT_DORSAL_TEST_CONFIG.copy()
        >>>     config['localization_model_tag'] = 'groundtruth'
        >>>     config['segmentation_model_tag'] = 'groundtruth'
        >>>     segmentations          = ibs.depc_annot.get('segmentation', aid_list, 'segmentations_img', config=config)
        >>>     refined_segmentations  = ibs.depc_annot.get('segmentation', aid_list, 'refined_segmentations_img', config=config)
        >>>     segmentation           = segmentations[0]
        >>>     refined_segmentation   = refined_segmentations[0]
        >>>     assert ut.hash_data(segmentation)         in ['owryieckgcmjqptjflybacfcmzgllhiw']
        >>>     assert ut.hash_data(refined_segmentation) in ['ddtxnvyvsskeazpftzlzbobfwxsfrvns']
        >>> finally:
        >>>     ibs.ibeis_plugin_curvrank_test_cleanup_groundtruth()

    Example3:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> config = DEFAULT_DORSAL_TEST_CONFIG.copy()
        >>> config['curvrank_model_type'] = 'dorsalfinfindrhybrid'
        >>> segmentations          = ibs.depc_annot.get('segmentation', aid_list, 'segmentations_img', config=config)
        >>> refined_segmentations  = ibs.depc_annot.get('segmentation', aid_list, 'refined_segmentations_img', config=config)
        >>> segmentation           = segmentations[0].tolist()
        >>> refined_segmentation   = refined_segmentations[0].tolist()
        >>> assert segmentation is None
        >>> assert refined_segmentation is None
    """
    ibs = depc.controller

    model_type        = config['curvrank_model_type']
    width             = config['curvrank_width']
    height            = config['curvrank_height']
    scale             = config['curvrank_scale']
    model_tag         = config['segmentation_model_tag']
    gt_radius         = config['segmentation_gt_radius']
    gt_opacity        = config['segmentation_gt_opacity']
    gt_smooth         = config['segmentation_gt_smooth']
    gt_smooth_margin  = config['segmentation_gt_smooth_margin']
    greyscale         = config['curvrank_greyscale']

    aid_list     = depc.get_ancestor_rowids('refinement',   refinement_rowid_list)
    refined_localizations = depc.get_native('refinement',   refinement_rowid_list,    'refined_img')
    refined_masks         = depc.get_native('refinement',   refinement_rowid_list,    'mask_img')
    pre_transforms        = depc.get_native('preprocess',   preprocess_rowid_list,    'pretransform')
    loc_transforms        = depc.get_native('localization', localization_rowid_list,  'transform')

    values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks,
                                                    pre_transforms, loc_transforms,
                                                    width=width, height=height,
                                                    scale=scale, model_type=model_type,
                                                    model_tag=model_tag,
                                                    groundtruth_radius=gt_radius,
                                                    groundtruth_opacity=gt_opacity,
                                                    groundtruth_smooth=gt_smooth,
                                                    groundtruth_smooth_margin=gt_smooth_margin,
                                                    greyscale=greyscale)
    segmentations, refined_segmentations = values

    for segmentation, refined_segmentation in zip(segmentations, refined_segmentations):
        if segmentation is None:
            segmentation_height, segmentation_width = 0, 0
        else:
            segmentation_height, segmentation_width = segmentation.shape[:2]
        if refined_segmentation is None:
            refined_segmentation_height, refined_segmentation_width = 0, 0
        else:
            refined_segmentation_height, refined_segmentation_width = refined_segmentation.shape[:2]

        yield (
            segmentation,
            segmentation_width,
            segmentation_height,
            refined_segmentation,
            refined_segmentation_width,
            refined_segmentation_height,
        )


class KeypointsConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('curvrank_model_type', 'dorsal'),
        ]


@register_preproc_annot(
    tablename='keypoints', parents=['segmentation', 'localization'],
    colnames=['success', 'start_y', 'start_x', 'end_y', 'end_x'],
    coltypes=[bool, int, int, int, int],
    configclass=KeypointsConfig,
    fname='curvrank_unoptimized',
    rm_extern_on_delete=True,
    chunksize=256,
)
# chunksize defines the max number of 'yield' below that will be called in a chunk
# so you would decrease chunksize on expensive calculations
def ibeis_plugin_curvrank_keypoints_depc(depc, segmentation_rowid_list, localization_rowid_list, config=None):
    r"""
    Refine localizations for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_keypoints_depc
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_keypoints_depc:0
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_keypoints_depc:1
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_keypoints_depc:2

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> values = ibs.depc_annot.get('keypoints', aid_list, None, config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> success, start_y, start_x, end_y, end_x = values[0]
        >>> assert success
        >>> assert (start_y, start_x) == (203, 3)
        >>> assert (end_y,   end_x)   == (198, 252)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> values = ibs.depc_annot.get('keypoints', aid_list, None, config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> success, start_y, start_x, end_y, end_x = values[0]
        >>> assert success
        >>> assert (start_y, start_x) == (56, 8)
        >>> assert (end_y,   end_x)   == (59, 358)

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> config = DEFAULT_DORSAL_TEST_CONFIG.copy()
        >>> config['curvrank_model_type'] = 'dorsalfinfindrhybrid'
        >>> values = ibs.depc_annot.get('keypoints', aid_list, None, config=config)
        >>> success, start_y, start_x, end_y, end_x = values[0]
        >>> assert success
        >>> assert (start_y, start_x) == (None, None)
        >>> assert (end_y,   end_x)   == (None, None)
    """
    ibs = depc.controller

    model_type = config['curvrank_model_type']

    segmentations   = depc.get_native('segmentation', segmentation_rowid_list, 'segmentations_img')
    localized_masks = depc.get_native('localization', localization_rowid_list, 'mask_img')

    values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks, model_type=model_type)
    success_list, starts, ends = values

    for success, start, end in zip(success_list, starts, ends):
        yield (
            success,
            None if start is None else start[0],
            None if start is None else start[1],
            None if end is None else end[0],
            None if end is None else end[1],
        )


class OutlineConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('curvrank_model_type',     'dorsal'),
            ut.ParamInfo('curvrank_scale',          DEFAULT_SCALE['dorsal']),
            ut.ParamInfo('outline_allow_diagonal',  False),
        ]


@register_preproc_annot(
    tablename='outline', parents=['segmentation', 'refinement', 'keypoints'],
    colnames=['success', 'outline'],
    coltypes=[bool, np.ndarray],
    configclass=OutlineConfig,
    fname='curvrank_unoptimized',
    rm_extern_on_delete=True,
    chunksize=256,
)
# chunksize defines the max number of 'yield' below that will be called in a chunk
# so you would decrease chunksize on expensive calculations
def ibeis_plugin_curvrank_outline_depc(depc, segmentation_rowid_list, refinement_rowid_list, keypoints_rowid_list, config=None):
    r"""
    Refine localizations for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_outline_depc
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_outline_depc:0
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_outline_depc:1
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_outline_depc:2
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_outline_depc:3

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> success_list = ibs.depc_annot.get('outline', aid_list, 'success', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> outlines = ibs.depc_annot.get('outline', aid_list, 'outline', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> outline = outlines[0]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(outline) in ['lyrkwgzncvjpjvovikkvspdkecardwyz']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> success_list = ibs.depc_annot.get('outline', aid_list, 'success', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> outlines = ibs.depc_annot.get('outline', aid_list, 'outline', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> outline = outlines[0]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(outline) in ['qqvetxfhhipfuqneuinwrvcztkjlfoak']

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list, part_rowid_list = ibs.ibeis_plugin_curvrank_test_setup_groundtruth()
        >>> try:
        >>>     config = DEFAULT_DORSAL_TEST_CONFIG.copy()
        >>>     config['localization_model_tag'] = 'groundtruth'
        >>>     config['segmentation_model_tag'] = 'groundtruth'
        >>>     success_list = ibs.depc_annot.get('outline', aid_list, 'success', config=config)
        >>>     outlines = ibs.depc_annot.get('outline', aid_list, 'outline', config=config)
        >>>     outline = outlines[0]
        >>>     assert success_list == [True]
        >>>     assert ut.hash_data(outline) in ['ykbndjqawiersnktufkmdtbwsfuexyeg']
        >>> finally:
        >>>     ibs.ibeis_plugin_curvrank_test_cleanup_groundtruth()

    Example3:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> config = DEFAULT_DORSAL_TEST_CONFIG.copy()
        >>> config['curvrank_model_type'] = 'dorsalfinfindrhybrid'
        >>> success_list = ibs.depc_annot.get('outline', aid_list, 'success', config=config)
        >>> outlines = ibs.depc_annot.get('outline', aid_list, 'outline', config=config)
        >>> outline = outlines[0]
        >>> assert success_list == [True]
        >>> assert outline is None
    """
    ibs = depc.controller

    success_list = depc.get_native('keypoints', keypoints_rowid_list, 'success')
    starts = get_zipped(depc, 'keypoints', keypoints_rowid_list, 'start_y', 'start_x')
    ends   = get_zipped(depc, 'keypoints', keypoints_rowid_list, 'end_y',   'end_x')
    refined_localizations = depc.get_native('refinement', refinement_rowid_list, 'refined_img')
    refined_masks         = depc.get_native('refinement', refinement_rowid_list, 'mask_img')
    refined_segmentations = depc.get_native('segmentation', segmentation_rowid_list, 'refined_segmentations_img')

    args = success_list, starts, ends, refined_localizations, refined_masks, refined_segmentations
    kwargs = {
        'model_type':     config['curvrank_model_type'],
        'scale':          config['curvrank_scale'],
        'allow_diagonal': config['outline_allow_diagonal'],
    }
    success_list, outlines = ibs.ibeis_plugin_curvrank_outline(*args, **kwargs)
    for success, outline in zip(success_list, outlines):
        yield (
            success,
            outline
        )


class TrailingEdgeConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('curvrank_model_type',                  'dorsal'),
            ut.ParamInfo('curvrank_width',                       DEFAULT_HEIGHT['dorsal']),
            ut.ParamInfo('curvrank_height',                      DEFAULT_WIDTH['dorsal']),
            ut.ParamInfo('curvrank_scale',                       DEFAULT_SCALE['dorsal']),
            ut.ParamInfo('trailing_edge_finfindr_smooth',        True, hideif=True),
            ut.ParamInfo('trailing_edge_finfindr_smooth_margin', 0.0, hideif=0.0),
        ]


@register_preproc_annot(
    tablename='trailing_edge', parents=['outline'],
    colnames=['success', 'trailing_edge'],
    coltypes=[bool, np.ndarray],
    configclass=TrailingEdgeConfig,
    fname='curvrank_unoptimized',
    rm_extern_on_delete=True,
    chunksize=256,
)
# chunksize defines the max number of 'yield' below that will be called in a chunk
# so you would decrease chunksize on expensive calculations
def ibeis_plugin_curvrank_trailing_edges_depc(depc, outline_rowid_list, config=None):
    r"""
    Refine localizations for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_trailing_edges_depc
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_trailing_edges_depc:0
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_trailing_edges_depc:1
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_trailing_edges_depc:2 --finfindr

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> success_list = ibs.depc_annot.get('trailing_edge', aid_list, 'success', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> trailing_edges = ibs.depc_annot.get('trailing_edge', aid_list, 'trailing_edge', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> trailing_edge = trailing_edges[0]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(trailing_edge) in ['wiabdtkbaqjuvszkyvyjnpomrivyadaa']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> outlines = ibs.depc_annot.get('outline', aid_list, 'outline', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> success_list = ibs.depc_annot.get('trailing_edge', aid_list, 'success', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> trailing_edges = ibs.depc_annot.get('trailing_edge', aid_list, 'trailing_edge', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> trailing_edge = trailing_edges[0]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(outlines) == ut.hash_data(trailing_edges)

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> config = DEFAULT_DORSAL_TEST_CONFIG.copy()
        >>> config['curvrank_model_type'] = 'dorsalfinfindrhybrid'
        >>> outlines = ibs.depc_annot.get('outline', aid_list, 'outline', config=config)
        >>> success_list = ibs.depc_annot.get('trailing_edge', aid_list, 'success', config=config)
        >>> trailing_edges = ibs.depc_annot.get('trailing_edge', aid_list, 'trailing_edge', config=config)
        >>> trailing_edge = trailing_edges[0]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(trailing_edge) in ['arkytzqvzttuthiaxbdvjuuxkuimetod']
    """
    ibs = depc.controller

    model_type              = config['curvrank_model_type']
    width                   = config['curvrank_width']
    height                  = config['curvrank_height']
    scale                   = config['curvrank_scale']
    finfindr_smooth         = config['trailing_edge_finfindr_smooth']
    finfindr_smooth_margin  = config['trailing_edge_finfindr_smooth_margin']

    aid_list = depc.get_ancestor_rowids('outline', outline_rowid_list)
    success_list      = depc.get_native('outline', outline_rowid_list, 'success')
    outlines          = depc.get_native('outline', outline_rowid_list, 'outline')

    values = ibs.ibeis_plugin_curvrank_trailing_edges(aid_list, success_list, outlines,
                                                      model_type=model_type, width=width,
                                                      height=height, scale=scale,
                                                      finfindr_smooth=finfindr_smooth,
                                                      finfindr_smooth_margin=finfindr_smooth_margin)
    success_list, trailing_edges = values

    for success, trailing_edge in zip(success_list, trailing_edges):
        yield (
            success,
            trailing_edge,
        )


class CurvatuveConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('curvature_scales',         DEFAULT_SCALES['dorsal']),
            ut.ParamInfo('curvatute_transpose_dims', False),
        ]


@register_preproc_annot(
    tablename='curvature', parents=['trailing_edge'],
    colnames=['success', 'curvature'],
    coltypes=[bool, np.ndarray],
    configclass=CurvatuveConfig,
    fname='curvrank_unoptimized',
    rm_extern_on_delete=True,
    chunksize=256,
)
# chunksize defines the max number of 'yield' below that will be called in a chunk
# so you would decrease chunksize on expensive calculations
def ibeis_plugin_curvrank_curvatures_depc(depc, trailing_edge_rowid_list, config=None):
    r"""
    Refine localizations for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_curvatures_depc
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_curvatures_depc:0
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_curvatures_depc:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> success_list = ibs.depc_annot.get('curvature', aid_list, 'success', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> curvatures = ibs.depc_annot.get('curvature', aid_list, 'curvature', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> curvature = curvatures[0]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(curvature) in ['yeyykrdbfxqyrbdumvpkvatjoddavdgn']
        >>> assert curvature.shape == (918, 4)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> success_list = ibs.depc_annot.get('curvature', aid_list, 'success', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> curvatures = ibs.depc_annot.get('curvature', aid_list, 'curvature', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> curvature = curvatures[0]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(curvature) in ['dpbusvatdgcdblmtwvodvlsnjuffdylp']
    """
    ibs = depc.controller

    scales         = config['curvature_scales']
    transpose_dims = config['curvatute_transpose_dims']

    success_list   = depc.get_native('trailing_edge', trailing_edge_rowid_list, 'success')
    trailing_edges = depc.get_native('trailing_edge', trailing_edge_rowid_list, 'trailing_edge')

    values = ibs.ibeis_plugin_curvrank_curvatures(success_list, trailing_edges,
                                                  scales=scales,
                                                  transpose_dims=transpose_dims)
    success_list, curvatures = values

    for success, curvature in zip(success_list, curvatures):
        yield (
            success,
            curvature,
        )


class CurvatuveDescriptorConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('curvature_scales',                   DEFAULT_SCALES['dorsal']),
            ut.ParamInfo('curvature_descriptor_curv_length',   1024),
            ut.ParamInfo('curvature_descriptor_num_keypoints', 32),
            ut.ParamInfo('curvature_descriptor_uniform',       False),
            ut.ParamInfo('curvature_descriptor_feat_dim',      32),
        ]


@register_preproc_annot(
    tablename='curvature_descriptor', parents=['curvature'],
    colnames=['success', 'descriptor'],
    coltypes=[bool, ('extern', ut.partial(ut.load_cPkl, verbose=False), ut.partial(ut.save_cPkl, verbose=False))],
    configclass=CurvatuveDescriptorConfig,
    fname='curvrank_unoptimized',
    rm_extern_on_delete=True,
    chunksize=256,
)
# chunksize defines the max number of 'yield' below that will be called in a chunk
# so you would decrease chunksize on expensive calculations
def ibeis_plugin_curvrank_curvature_descriptors_depc(depc, curvature_rowid_list, config=None):
    r"""
    Refine localizations for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_curvature_descriptors_depc
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_curvature_descriptors_depc:0
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_curvature_descriptors_depc:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> success_list = ibs.depc_annot.get('curvature_descriptor', aid_list, 'success', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> curvature_descriptor_dicts = ibs.depc_annot.get('curvature_descriptor', aid_list, 'descriptor', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> curvature_descriptor_dict = curvature_descriptor_dicts[0]
        >>> assert success_list == [True]
        >>> hash_list = [
        >>>     ut.hash_data(curvature_descriptor_dict[scale])
        >>>     for scale in sorted(list(curvature_descriptor_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['mkhgqrrkhisuaenxkuxgbbcqpdfpoofp']
        >>> assert curvature_descriptor_dict['0.0400'].shape == (496, 32)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> success_list = ibs.depc_annot.get('curvature_descriptor', aid_list, 'success', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> curvature_descriptor_dicts = ibs.depc_annot.get('curvature_descriptor', aid_list, 'descriptor', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> curvature_descriptor_dict = curvature_descriptor_dicts[0]
        >>> assert success_list == [True]
        >>> hash_list = [
        >>>     ut.hash_data(curvature_descriptor_dict[scale])
        >>>     for scale in sorted(list(curvature_descriptor_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['zacdsfedcywqdyqozfhdirrcqnypaazw']
    """
    ibs = depc.controller

    scales         = config['curvature_scales']
    curv_length    = config['curvature_descriptor_curv_length']
    num_keypoints  = config['curvature_descriptor_num_keypoints']
    uniform        = config['curvature_descriptor_uniform']
    feat_dim       = config['curvature_descriptor_feat_dim']

    success_list = depc.get_native('curvature', curvature_rowid_list, 'success')
    curvatures   = depc.get_native('curvature', curvature_rowid_list, 'curvature')

    values = ibs.ibeis_plugin_curvrank_curvature_descriptors(
        success_list,
        curvatures,
        curv_length,
        scales,
        num_keypoints,
        uniform,
        feat_dim
    )
    success_list, curvature_descriptor_dicts = values

    for success, curvature_descriptor_dict in zip(success_list, curvature_descriptor_dicts):
        yield (
            success,
            curvature_descriptor_dict,
        )


class CurvatuveDescriptorOptimizedConfig(dtool.Config):
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
            ut.ParamInfo('curvrank_model_type',                  'dorsal'),
            ut.ParamInfo('curvrank_width',                       DEFAULT_WIDTH['dorsal']),
            ut.ParamInfo('curvrank_height',                      DEFAULT_HEIGHT['dorsal']),
            ut.ParamInfo('curvrank_greyscale',                   False),
            ut.ParamInfo('curvrank_scale',                       DEFAULT_SCALE['dorsal']),
            ut.ParamInfo('curvature_scales',                     DEFAULT_SCALES['dorsal']),
            ut.ParamInfo('outline_allow_diagonal',               DEFAULT_ALLOW_DIAGONAL['dorsal']),
            ut.ParamInfo('curvatute_transpose_dims',             DEFAULT_TRANSPOSE_DIMS['dorsal']),
            ut.ParamInfo('localization_model_tag',               'localization'),
            ut.ParamInfo('segmentation_model_tag',               'segmentation'),
            ut.ParamInfo('segmentation_gt_radius',               25),
            ut.ParamInfo('segmentation_gt_opacity',              0.5),
            ut.ParamInfo('segmentation_gt_smooth',               True),
            ut.ParamInfo('segmentation_gt_smooth_margin',        0.001),
            ut.ParamInfo('trailing_edge_finfindr_smooth',        True, hideif=True),
            ut.ParamInfo('trailing_edge_finfindr_smooth_margin', 0.0, hideif=0.0),
            ut.ParamInfo('curvature_descriptor_curv_length',     1024),
            ut.ParamInfo('curvature_descriptor_num_keypoints',   32),
            ut.ParamInfo('curvature_descriptor_uniform',         False),
            ut.ParamInfo('curvature_descriptor_feat_dim',        32),
        ]

        return param_list


@register_preproc_annot(
    tablename='curvature_descriptor_optimized', parents=[ROOT],
    colnames=['success', 'descriptor'],
    coltypes=[bool, ('extern', ut.partial(ut.load_cPkl, verbose=False), ut.partial(ut.save_cPkl, verbose=False))],
    configclass=CurvatuveDescriptorOptimizedConfig,
    fname='curvrank_optimized',
    rm_extern_on_delete=True,
    chunksize=256,
)
# chunksize defines the max number of 'yield' below that will be called in a chunk
# so you would decrease chunksize on expensive calculations
def ibeis_plugin_curvrank_curvature_descriptors_optimized_depc(depc, aid_list, config=None):
    r"""
    Refine localizations for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_curvature_descriptors_optimized_depc
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_curvature_descriptors_optimized_depc:0
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_curvature_descriptors_optimized_depc:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> success_list = ibs.depc_annot.get('curvature_descriptor_optimized', aid_list, 'success', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> curvature_descriptor_dicts = ibs.depc_annot.get('curvature_descriptor', aid_list, 'descriptor', config=DEFAULT_DORSAL_TEST_CONFIG)
        >>> curvature_descriptor_dict = curvature_descriptor_dicts[0]
        >>> assert success_list == [True]
        >>> hash_list = [
        >>>     ut.hash_data(curvature_descriptor_dict[scale])
        >>>     for scale in sorted(list(curvature_descriptor_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['mkhgqrrkhisuaenxkuxgbbcqpdfpoofp']
        >>> assert curvature_descriptor_dict['0.0400'].shape == (496, 32)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> success_list = ibs.depc_annot.get('curvature_descriptor_optimized', aid_list, 'success', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> curvature_descriptor_dicts = ibs.depc_annot.get('curvature_descriptor', aid_list, 'descriptor', config=DEFAULT_FLUKE_TEST_CONFIG)
        >>> curvature_descriptor_dict = curvature_descriptor_dicts[0]
        >>> assert success_list == [True]
        >>> hash_list = [
        >>>     ut.hash_data(curvature_descriptor_dict[scale])
        >>>     for scale in sorted(list(curvature_descriptor_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['zacdsfedcywqdyqozfhdirrcqnypaazw']
    """
    ibs = depc.controller

    config_ = _convert_depc_config_to_kwargs_config(config)
    values = ibs.ibeis_plugin_curvrank_pipeline_compute(aid_list, config_)
    success_list, curvature_descriptor_dicts = values

    for success, curvature_descriptor_dict in zip(success_list, curvature_descriptor_dicts):
        yield (
            success,
            curvature_descriptor_dict,
        )


@register_ibs_method
def ibeis_plugin_curvrank_scores_depc(ibs, db_aid_list, qr_aid_list, **kwargs):
    r"""
    CurvRank Example

    Args:
        ibs       (IBEISController): IBEIS controller object
        lnbnn_k   (int): list of image rowids (aids)

    Returns:
        score_dict

    CommandLine:
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_scores_depc
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_scores_depc:0
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_scores_depc:1
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_scores_depc:2
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_scores_depc:3

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> db_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Dorsal Database')
        >>> db_aid_list = ibs.get_imageset_aids(db_imageset_rowid)
        >>> qr_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Dorsal Query')
        >>> qr_aid_list = ibs.get_imageset_aids(qr_imageset_rowid)
        >>> config = DEFAULT_DORSAL_TEST_CONFIG
        >>> config['curvrank_daily_cache'] = False
        >>> score_dict_iter = ibs.ibeis_plugin_curvrank_scores_depc(db_aid_list, [qr_aid_list], config=config, use_depc_optimized=False)
        >>> score_dict_list = list(score_dict_iter)
        >>> qr_aid_list, score_dict = score_dict_list[0]
        >>> for key in score_dict:
        >>>     score_dict[key] = round(score_dict[key], 8)
        >>> result = score_dict
        >>> print(result)
        {1: -31.81339289, 2: -3.7092349, 3: -4.95274189}

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> db_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Dorsal Database')
        >>> db_aid_list = ibs.get_imageset_aids(db_imageset_rowid)
        >>> qr_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Dorsal Query')
        >>> qr_aid_list = ibs.get_imageset_aids(qr_imageset_rowid)
        >>> config = DEFAULT_DORSAL_TEST_CONFIG
        >>> config['curvrank_daily_cache'] = False
        >>> score_dict_iter = ibs.ibeis_plugin_curvrank_scores_depc(db_aid_list, [qr_aid_list], config=config, use_depc_optimized=True)
        >>> score_dict_list = list(score_dict_iter)
        >>> qr_aid_list, score_dict = score_dict_list[0]
        >>> for key in score_dict:
        >>>     score_dict[key] = round(score_dict[key], 8)
        >>> result = score_dict
        >>> print(result)
        {1: -31.81339289, 2: -3.7092349, 3: -4.95274189}

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> db_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Fluke Database')
        >>> db_aid_list = ibs.get_imageset_aids(db_imageset_rowid)
        >>> qr_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Fluke Query')
        >>> qr_aid_list = ibs.get_imageset_aids(qr_imageset_rowid)
        >>> config = DEFAULT_FLUKE_TEST_CONFIG
        >>> config['curvrank_daily_cache'] = False
        >>> score_dict_iter = ibs.ibeis_plugin_curvrank_scores_depc(db_aid_list, [qr_aid_list], config=config, use_depc_optimized=False)
        >>> score_dict_list = list(score_dict_iter)
        >>> qr_aid_list, score_dict = score_dict_list[0]
        >>> for key in score_dict:
        >>>     score_dict[key] = round(score_dict[key], 8)
        >>> result = score_dict
        >>> print(result)
        {14: -1.00862974, 7: -0.55433992, 8: -0.70058628, 9: -0.3044969, 10: -0.27739539, 11: -7.8684881, 12: -1.01431028, 13: -1.46861451}

    Example3:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> db_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Fluke Database')
        >>> db_aid_list = ibs.get_imageset_aids(db_imageset_rowid)
        >>> qr_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Fluke Query')
        >>> qr_aid_list = ibs.get_imageset_aids(qr_imageset_rowid)
        >>> config = DEFAULT_FLUKE_TEST_CONFIG
        >>> config['curvrank_daily_cache'] = False
        >>> score_dict_iter = ibs.ibeis_plugin_curvrank_scores_depc(db_aid_list, [qr_aid_list], config=config, use_depc_optimized=True)
        >>> score_dict_list = list(score_dict_iter)
        >>> qr_aid_list, score_dict = score_dict_list[0]
        >>> for key in score_dict:
        >>>     score_dict[key] = round(score_dict[key], 8)
        >>> result = score_dict
        >>> print(result)
        {14: -1.00862974, 7: -0.55433992, 8: -0.70058628, 9: -0.3044969, 10: -0.27739539, 11: -7.8684881, 12: -1.01431028, 13: -1.46861451}
    """
    kwargs['use_depc'] = True
    kwargs['config'] = _convert_depc_config_to_kwargs_config(kwargs.get('config', {}))
    return ibs.ibeis_plugin_curvrank_scores(db_aid_list, qr_aid_list, **kwargs)


def get_match_results(depc, qaid_list, daid_list, score_list, config):
    """ converts table results into format for ipython notebook """
    #qaid_list, daid_list = request.get_parent_rowids()
    #score_list = request.score_list
    #config = request.config

    unique_qaids, groupxs = ut.group_indices(qaid_list)
    #grouped_qaids_list = ut.apply_grouping(qaid_list, groupxs)
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

        is_valid = (daid_list_ != qaid)
        daid_list_ = daid_list_.compress(is_valid)
        dnid_list_ = dnid_list_.compress(is_valid)
        annot_scores = annot_scores.compress(is_valid)

        # Hacked in version of creating an annot match object
        match_result = ibeis.AnnotMatch()
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

    def overlay_trailing_edge(request, chip, outline, trailing_edge, edge_color=(0, 255, 255)):
        import cv2
        scale = request.config.curvrank_scale

        chip_ = np.copy(chip)
        chip_ = cv2.resize(chip_, dsize=None, fx=scale, fy=scale)
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

        chips = depc.get('localization', aid_list, 'localized_img', config=request.config)
        outlines = [None] * len(chips)
        trailing_edges = [None] * len(chips)

        model_type = request.config.curvrank_model_type
        if model_type in ['dorsalfinfindrhybrid']:
            chips = depc.get('localization', aid_list, 'localized_img', config=DEFAULT_DORSAL_TEST_CONFIG)

        if overlay:
            if model_type not in ['dorsalfinfindrhybrid']:
                outlines = depc.get('outline', aid_list, 'outline', config=request.config)
            trailing_edges = depc.get('trailing_edge', aid_list, 'trailing_edge', config=request.config)

        overlay_chips = [
            request.overlay_trailing_edge(chip, outline, trailing_edge)
            for chip, outline, trailing_edge in zip(chips, outlines, trailing_edges)
        ]
        return overlay_chips

    def render_single_result(request, cm, aid, **kwargs):
        # HACK FOR WEB VIEWER
        overlay = kwargs.get('draw_fmatches')
        chips = request.get_fmatch_overlayed_chip([cm.qaid, aid], overlay=overlay,
                                                  config=request.config)
        import vtool as vt
        out_img = vt.stack_image_list(chips)
        return out_img

    def postprocess_execute(request, parent_rowids, result_list):
        qaid_list, daid_list = list(zip(*parent_rowids))
        score_list = ut.take_column(result_list, 0)
        depc = request.depc
        config = request.config
        cm_list = list(get_match_results(depc, qaid_list, daid_list,
                                         score_list, config))
        return cm_list

    def execute(request, *args, **kwargs):
        kwargs['use_cache'] = False
        result_list = super(CurvRankRequest, request).execute(*args, **kwargs)
        qaids = kwargs.pop('qaids', None)
        if qaids is not None:
            result_list = [
                result for result in result_list
                if result.qaid in qaids
            ]
        return result_list


class CurvRankDorsalConfig(dtool.Config):  # NOQA
    """
    CommandLine:
        python -m ibeis_curvrank._plugin_depc --test-CurvRankDorsalConfig

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> config = CurvRankDorsalConfig()
        >>> result = config.get_cfgstr()
        >>> print(result)
        CurvRankDorsal(curvature_descriptor_curv_length=1024,curvature_descriptor_feat_dim=32,curvature_descriptor_num_keypoints=32,curvature_descriptor_uniform=False,curvature_scales=[0.04 0.06 0.08 0.1 ],curvatute_transpose_dims=False,curvrank_cache_recompute=False,curvrank_daily_cache=True,curvrank_daily_tag=global,curvrank_height=256,curvrank_model_type=dorsal,curvrank_scale=4,curvrank_width=256,localization_model_tag=localization,outline_allow_diagonal=False,segmentation_gt_opacity=0.5,segmentation_gt_radius=25,segmentation_gt_smooth=True,segmentation_gt_smooth_margin=0.001,segmentation_model_tag=segmentation)
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
    _tablename = 'CurvRankDorsal'


@register_preproc_annot(
    tablename='CurvRankDorsal', parents=[ROOT, ROOT],
    colnames=['score'], coltypes=[float],
    configclass=CurvRankDorsalConfig,
    requestclass=CurvRankDorsalRequest,
    fname='curvrank_scores_dorsal',
    rm_extern_on_delete=True,
    chunksize=None)
def ibeis_plugin_curvrank_dorsal(depc, qaid_list, daid_list, config):
    r"""
    CommandLine:
        python -m ibeis_curvrank._plugin_depc --exec-ibeis_plugin_curvrank_dorsal --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> import itertools as it
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
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
        >>> prop_list = depc.get('CurvRankDorsal', root_rowids)
        >>> # Call function normally
        >>> score_list = list(ibeis_plugin_curvrank_dorsal(depc, qaid_list, daid_list, config))
        >>> am_list2 = list(get_match_results(depc, qaid_list, daid_list, score_list, config))
        >>> assert score_list == prop_list, 'error in cache'
        >>> assert np.all(am_list1[0].score_list == am_list2[0].score_list)
        >>> ut.quit_if_noshow()
        >>> am = am_list2[0]
        >>> am.ishow_analysis(request)
        >>> ut.show_if_requested()
    """
    ibs = depc.controller

    label = 'CurvRankDorsal'
    value_iter = ibs.ibeis_plugin_curvrank(label, qaid_list, daid_list, config)
    for value in value_iter:
        yield value


class CurvRankFlukeConfig(dtool.Config):  # NOQA
    """
    CommandLine:
        python -m ibeis_curvrank._plugin_depc --test-CurvRankFlukeConfig

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> config = CurvRankFlukeConfig()
        >>> result = config.get_cfgstr()
        >>> print(result)
        CurvRankFluke(curvature_descriptor_curv_length=1024,curvature_descriptor_feat_dim=32,curvature_descriptor_num_keypoints=32,curvature_descriptor_uniform=False,curvature_scales=[0.02 0.04 0.06 0.08],curvatute_transpose_dims=True,curvrank_cache_recompute=False,curvrank_daily_cache=True,curvrank_daily_tag=global,curvrank_height=192,curvrank_model_type=fluke,curvrank_scale=3,curvrank_width=384,localization_model_tag=localization,outline_allow_diagonal=True,segmentation_gt_opacity=0.5,segmentation_gt_radius=25,segmentation_gt_smooth=True,segmentation_gt_smooth_margin=0.001,segmentation_model_tag=segmentation)
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
    _tablename = 'CurvRankFluke'


@register_preproc_annot(
    tablename='CurvRankFluke', parents=[ROOT, ROOT],
    colnames=['score'], coltypes=[float],
    configclass=CurvRankFlukeConfig,
    requestclass=CurvRankFlukeRequest,
    fname='curvrank_scores_fluke',
    rm_extern_on_delete=True,
    chunksize=None)
def ibeis_plugin_curvrank_fluke(depc, qaid_list, daid_list, config):
    r"""
    CommandLine:
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_fluke --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> import itertools as it
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
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
        >>> prop_list = depc.get('CurvRankFluke', root_rowids)
        >>> # Call function normally
        >>> score_list = list(ibeis_plugin_curvrank_fluke(depc, qaid_list, daid_list, config))
        >>> am_list2 = list(get_match_results(depc, qaid_list, daid_list, score_list, config))
        >>> assert score_list == prop_list, 'error in cache'
        >>> assert np.all(am_list1[0].score_list == am_list2[0].score_list)
        >>> ut.quit_if_noshow()
        >>> am_list2 = [am for am in am_list2 if am.qaid == 23]
        >>> am = am_list2[0]
        >>> am.ishow_analysis(request)
        >>> ut.show_if_requested()
    """
    ibs = depc.controller

    label = 'CurvRankFluke'
    value_iter = ibs.ibeis_plugin_curvrank(label, qaid_list, daid_list, config)
    for value in value_iter:
        yield value


class CurvRankFinfindrHybridDorsalConfig(dtool.Config):  # NOQA
    """
    CommandLine:
        python -m ibeis_curvrank._plugin_depc --test-CurvRankFinfindrHybridDorsalConfig

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> config = CurvRankFinfindrHybridDorsalConfig()
        >>> result = config.get_cfgstr()
        >>> print(result)
        CurvRankFinfindrHybridDorsal(curvature_descriptor_curv_length=1024,curvature_descriptor_feat_dim=32,curvature_descriptor_num_keypoints=32,curvature_descriptor_uniform=False,curvature_scales=[0.04 0.06 0.08 0.1 ],curvatute_transpose_dims=False,curvrank_cache_recompute=False,curvrank_daily_cache=True,curvrank_daily_tag=global,curvrank_greyscale=False,curvrank_height=256,curvrank_model_type=dorsalfinfindrhybrid,curvrank_scale=4,curvrank_width=256,localization_model_tag=localization,outline_allow_diagonal=False,segmentation_gt_opacity=0.5,segmentation_gt_radius=25,segmentation_gt_smooth=True,segmentation_gt_smooth_margin=0.001,segmentation_model_tag=segmentation)
    """
    def get_param_info_list(self):
        value_mapping_dict = {
            'curvrank_model_type'                : 'dorsalfinfindrhybrid',
            'curvature_descriptor_uniform'       : True,
            # 'curvature_descriptor_num_keypoints' : 64,
        }
        param_list = []
        key_list = DEFAULT_DORSAL_TEST_CONFIG.keys()
        for key in sorted(key_list):
            value = DEFAULT_DORSAL_TEST_CONFIG[key]
            value = value_mapping_dict.get(key, value)
            if key.startswith('trailing_edge_finfindr_') or key.startswith('index_'):
                param = ut.ParamInfo(key, value, hideif=value)
            else:
                param = ut.ParamInfo(key, value)
            param_list.append(param)
        return param_list


class CurvRankFinfindrHybridDorsalRequest(CurvRankRequest):  # NOQA
    _tablename = 'CurvRankFinfindrHybridDorsal'


@register_preproc_annot(
    tablename='CurvRankFinfindrHybridDorsal', parents=[ROOT, ROOT],
    colnames=['score'], coltypes=[float],
    configclass=CurvRankFinfindrHybridDorsalConfig,
    requestclass=CurvRankFinfindrHybridDorsalRequest,
    fname='curvrank_scores_dorsal',
    rm_extern_on_delete=True,
    chunksize=None)
def ibeis_plugin_curvrank_finfindr_hybrid_dorsal(depc, qaid_list, daid_list, config):
    r"""
    CommandLine:
        python -m ibeis_curvrank._plugin_depc --exec-ibeis_plugin_curvrank_dorsal --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> import itertools as it
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> depc = ibs.depc_annot
        >>> imageset_rowid_list = ibs.get_imageset_imgsetids_from_text(['Dorsal Database', 'Dorsal Query'])
        >>> aid_list = list(set(ut.flatten(ibs.get_imageset_aids(imageset_rowid_list))))
        >>> root_rowids = tuple(zip(*it.product(aid_list, aid_list)))
        >>> qaid_list, daid_list = root_rowids
        >>> config = CurvRankFinfindrHybridDorsalConfig()
        >>> # Call function via request
        >>> request = CurvRankFinfindrHybridDorsalRequest.new(depc, aid_list, aid_list)
        >>> am_list1 = request.execute()
        >>> # Call function via depcache
        >>> prop_list = depc.get('CurvRankFinfindrHybridDorsal', root_rowids)
        >>> # Call function normally
        >>> score_list = list(ibeis_plugin_curvrank_dorsal(depc, qaid_list, daid_list, config))
        >>> am_list2 = list(get_match_results(depc, qaid_list, daid_list, score_list, config))
        >>> assert score_list == prop_list, 'error in cache'
        >>> assert np.all(am_list1[0].score_list == am_list2[0].score_list)
        >>> ut.quit_if_noshow()
        >>> am = am_list2[0]
        >>> am.ishow_analysis(request)
        >>> ut.show_if_requested()
    """
    ibs = depc.controller

    label = 'CurvRankFinfindrHybridDorsal'
    value_iter = ibs.ibeis_plugin_curvrank(label, qaid_list, daid_list, config)
    for value in value_iter:
        yield value


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis_curvrank._plugin_depc --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
