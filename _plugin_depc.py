from __future__ import absolute_import, division, print_function
from ibeis.control import controller_inject  # NOQA
import numpy as np
import utool as ut
import vtool as vt
import dtool


register_preproc_image = controller_inject.register_preprocs['image']
register_preproc_annot = controller_inject.register_preprocs['annot']


DEFAULT_WIDTH  = 256
DEFAULT_HEIGHT = 256
DEFAULT_SCALE  = 4
DEFAULT_SCALES = np.array([0.04, 0.06, 0.08, 0.10], dtype=np.float32)


DEFAULT_TEST_CONFIG = {
    'curvrank_height'                   : DEFAULT_WIDTH,
    'curvrank_width'                    : DEFAULT_HEIGHT,
    'curvrank_scale'                    : DEFAULT_SCALE,
    'localization_model_tag'            : 'localization',
    'segmentation_model_tag'            : 'segmentation',
    'outline_allow_diagonal'            : False,
    'curvature_scales'                  : DEFAULT_SCALES,
    'curvatute_transpose_dims'          : False,
    'curvature_descriptor_curv_length'  : 1024,
    'curvature_descriptor_num_keypoints': 32,
    'curvature_descriptor_uniform'      : False,
    'curvature_descriptor_feat_dim'     : 32,
}


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


class PreprocessConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('curvrank_height', 256),
            ut.ParamInfo('curvrank_width',  256),
            ut.ParamInfo('ext', '.npy', hideif='.npy'),
        ]


@register_preproc_image(
    tablename='preprocess', parents=['images'],
    colnames=['resized_img', 'resized_width', 'resized_height', 'mask_img', 'mask_width', 'mask_height', 'pretransform'],
    coltypes=[('extern', np.load, np.save), int, int, ('extern', np.load, np.save), int, int, np.ndarray],
    configclass=PreprocessConfig,
    fname='curvrank',
    rm_extern_on_delete=True,
    chunksize=256,
)
def ibeis_plugin_curvrank_preprocessing_depc(depc, gid_list, config=None):
    r"""
    Pre-process images for CurvRank with Dependency Cache (depc)

    Args:
        depc      (Dependency Cache): IBEIS dependency cache object
        gid_list  (list of int): list of image rowids (gids)
        config    (PreprocessConfig): config for depcache

    CommandLine:
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_preprocessing_depc

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> resized_images = ibs.depc_image.get('preprocess', gid_list, 'resized_img',  config=DEFAULT_TEST_CONFIG)
        >>> resized_masks  = ibs.depc_image.get('preprocess', gid_list, 'mask_img',     config=DEFAULT_TEST_CONFIG)
        >>> pre_transforms = ibs.depc_image.get('preprocess', gid_list, 'pretransform', config=DEFAULT_TEST_CONFIG)
        >>> resized_image = resized_images[0]
        >>> resized_mask  = resized_masks[0]
        >>> pre_transform = pre_transforms[0]
        >>> assert ut.hash_data(resized_image) in ['ynbsgqgfutslspmatpenvcbtgedsyzoo']
        >>> assert ut.hash_data(resized_mask)  in ['mnhartnytowmmhskblocubqmzhbofynr']
        >>> result = pre_transform
        >>> print(result)
        [[ 0.11077456  0.          0.        ]
         [ 0.          0.11077456 38.        ]
         [ 0.          0.          1.        ]]
    """
    ibs = depc.controller

    width  = config['curvrank_width']
    height = config['curvrank_height']

    values = ibs.ibeis_plugin_curvrank_preprocessing(gid_list, width=width, height=height)
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
            ut.ParamInfo('curvrank_height', 256),
            ut.ParamInfo('curvrank_width',  256),
            ut.ParamInfo('localization_model_tag', 'localization'),
            ut.ParamInfo('ext', '.npy', hideif='.npy'),
        ]


@register_preproc_image(
    tablename='localization', parents=['preprocess'],
    colnames=['localized_img', 'localized_width', 'localized_height', 'mask_img', 'mask_width', 'mask_height', 'transform'],
    coltypes=[('extern', np.load, np.save), int, int, ('extern', np.load, np.save), int, int, np.ndarray],
    configclass=LocalizationConfig,
    fname='curvrank',
    rm_extern_on_delete=True,
    chunksize=256,
)
# chunksize defines the max number of 'yield' below that will be called in a chunk
# so you would decrease chunksize on expensive calculations
def ibeis_plugin_curvrank_localization_depc(depc, preprocess_rowid_list, config=None):
    r"""
    Localize images for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_localization_depc

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> localized_images = ibs.depc_image.get('localization', gid_list, 'localized_img',  config=DEFAULT_TEST_CONFIG)
        >>> localized_masks  = ibs.depc_image.get('localization', gid_list, 'mask_img',     config=DEFAULT_TEST_CONFIG)
        >>> loc_transforms = ibs.depc_image.get('localization', gid_list, 'transform', config=DEFAULT_TEST_CONFIG)
        >>> localized_image = localized_images[0]
        >>> localized_mask  = localized_masks[0]
        >>> loc_transform = loc_transforms[0]
        >>> # localized_image appears to differ very slightly in ubuntu vs. mac. Hashes below for each respectively.
        >>> #TODO verify that mac/ubuntu values are consistent on those OSes
        >>> assert ut.hash_data(localized_image) in ['mbwtvdojxaidtmcrqvyamkgpchzupfsh','pbgpmewfannhnrsrfxixdnhwczbkordr']
        >>> assert ut.hash_data(localized_mask)  in ['pzzhgfsbhcsayowiwusjjekzlxaxbrpu']
        >>> # for above reasons, some voodoo to compare loc_transform
        >>> loc_transform_ubuntu = np.array([[ 0.63338047,  0.12626281, -0.11245003],
        >>>                                  [-0.12531438,  0.63420326, -0.00189855],
        >>>                                  [ 0.        ,  0.        ,  1.        ]])
        >>> assert np.all(np.abs(loc_transform - loc_transform_ubuntu) < 1e-6)
    """
    ibs = depc.controller

    width     = config['curvrank_width']
    height    = config['curvrank_height']
    model_tag = config['localization_model_tag']

    resized_images = depc.get_native('preprocess', preprocess_rowid_list, 'resized_img')
    resized_masks  = depc.get_native('preprocess', preprocess_rowid_list, 'mask_img')

    values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks,
                                                    width=width, height=height,
                                                    model_tag=model_tag)
    localized_images, localized_masks, loc_transforms = values

    # yield each column defined in register_preproc_image
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
            ut.ParamInfo('curvrank_width',  256),
            ut.ParamInfo('curvrank_height', 256),
            ut.ParamInfo('curvrank_scale',  4),
            ut.ParamInfo('ext', '.npy', hideif='.npy'),
        ]


@register_preproc_image(
    tablename='refinement', parents=['localization', 'preprocess'],
    colnames=['refined_img', 'refined_width', 'refined_height', 'mask_img', 'mask_width', 'mask_height'],
    coltypes=[('extern', np.load, np.save), int, int, ('extern', np.load, np.save), int, int],
    configclass=RefinementConfig,
    fname='curvrank',
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

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> refined_localizations = ibs.depc_image.get('refinement', gid_list, 'refined_img', config=DEFAULT_TEST_CONFIG)
        >>> refined_masks         = ibs.depc_image.get('refinement', gid_list, 'mask_img', config=DEFAULT_TEST_CONFIG)
        >>> refined_localization  = refined_localizations[0]
        >>> refined_mask          = refined_masks[0]
        >>> #TODO verify that mac/ubuntu values are consistent on those OSes
        >>> # why are these values different than in above? have we cached bad stuff? I'm guessing yes.
        >>> assert ut.hash_data(refined_localization) in ['glgopopgyjfuscigvpudxzcjvgvxpoef', 'idspzbmvqxvgoyyjkuseeztpmjkbisrz']
        >>> assert ut.hash_data(refined_mask)         in ['yozbarldhrafcksnimwxhgsnmfochjnv', 'luqzalptfdneljbkslrpufypwmajsmdv']
    """
    ibs = depc.controller

    width  = config['curvrank_width']
    height = config['curvrank_height']
    scale  = config['curvrank_scale']

    gid_list = depc.get_ancestor_rowids('preprocess',  preprocess_rowid_list)
    loc_transforms   = depc.get_native('localization', localization_rowid_list, 'transform')
    pre_transforms   = depc.get_native('preprocess',   preprocess_rowid_list,   'pretransform')

    values = ibs.ibeis_plugin_curvrank_refinement(gid_list, pre_transforms, loc_transforms,
                                                  width=width, height=height, scale=scale)
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
            ut.ParamInfo('curvrank_width',  256),
            ut.ParamInfo('curvrank_height', 256),
            ut.ParamInfo('curvrank_scale',  4),
            ut.ParamInfo('segmentation_model_tag', 'segmentation'),
            ut.ParamInfo('ext', '.npy', hideif='.npy'),
        ]


@register_preproc_image(
    tablename='segmentation', parents=['refinement'],
    colnames=['segmentations_img', 'refined_width', 'refined_height', 'refined_segmentations_img', 'refined_segmentations_width', 'refined_segmentations_height'],
    coltypes=[('extern', np.load, np.save), int, int, ('extern', np.load, np.save), int, int],
    configclass=SegmentationConfig,
    fname='curvrank',
    rm_extern_on_delete=True,
    chunksize=256,
)
# chunksize defines the max number of 'yield' below that will be called in a chunk
# so you would decrease chunksize on expensive calculations
def ibeis_plugin_curvrank_segmentation_depc(depc, refinement_rowid_list, config=None):
    r"""
    Refine localizations for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_segmentation_depc

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> segmentations          = ibs.depc_image.get('segmentation', gid_list, 'segmentations_img', config=DEFAULT_TEST_CONFIG)
        >>> refined_segmentations  = ibs.depc_image.get('segmentation', gid_list, 'refined_segmentations_img', config=DEFAULT_TEST_CONFIG)
        >>> segmentation           = segmentations[0]
        >>> refined_segmentation   = refined_segmentations[0]
        >>> assert ut.hash_data(segmentation)         in ['vbmvokttgelinljiiqbmhhxehgcwnjxe', 'wnfimwthormmytbumjnqrhjbsfjccksy']
        >>> assert ut.hash_data(refined_segmentation) in ['hrcdfxsblmgzkmkrywytxurpkxyeyhyg', 'fmmuefyrgmpyaaeakqnbgbafrhwbvohf']
    """
    ibs = depc.controller

    width     = config['curvrank_width']
    height    = config['curvrank_height']
    scale     = config['curvrank_scale']
    model_tag = config['segmentation_model_tag']

    refined_localizations = depc.get_native('refinement', refinement_rowid_list, 'refined_img')
    refined_masks         = depc.get_native('refinement', refinement_rowid_list, 'mask_img')

    values = ibs.ibeis_plugin_curvrank_segmentation(refined_localizations, refined_masks,
                                                    width=width, height=height, scale=scale,
                                                    model_tag=model_tag)
    segmentations, refined_segmentations = values

    for segmentation, refined_segmentation in zip(segmentations, refined_segmentations):
        segmentation_height, segmentation_width = segmentation.shape[:2]
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
        return []


@register_preproc_image(
    tablename='keypoints', parents=['segmentation', 'localization'],
    colnames=['success', 'start_y', 'start_x', 'end_y', 'end_x'],
    coltypes=[bool, int, int, int, int],
    configclass=KeypointsConfig,
    fname='curvrank',
    rm_extern_on_delete=True,
    chunksize=256,
)
# chunksize defines the max number of 'yield' below that will be called in a chunk
# so you would decrease chunksize on expensive calculations
def ibeis_plugin_curvrank_keypoints_depc(depc, segmentation_rowid_list, localization_rowid_list, config=None):
    r"""
    Refine localizations for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m ibeis_curvrank._plugin_depc --test-ibeis_plugin_curvrank_segmentation_depc

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> values = ibs.depc_image.get('keypoints', gid_list, None, config=DEFAULT_TEST_CONFIG)
        >>> success, start_y, start_x, end_y, end_x = values[0]
        >>> assert success
        >>> assert (start_y, start_x) == (204, 1)
        >>> assert (end_y,   end_x)   == (199, 251)
    """
    ibs = depc.controller

    segmentations   = depc.get_native('segmentation', segmentation_rowid_list, 'segmentations_img')
    localized_masks = depc.get_native('localization', localization_rowid_list, 'mask_img')

    values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks)
    success_list, starts, ends = values

    for success, start, end in zip(success_list, starts, ends):
        yield (
            success,
            start[0],
            start[1],
            end[0],
            end[1]
        )


class OutlineConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('curvrank_scale',  4),
            ut.ParamInfo('outline_allow_diagonal', False),
        ]


@register_preproc_image(
    tablename='outline', parents=['segmentation', 'refinement', 'keypoints'],
    colnames=['success', 'outline'],
    coltypes=[bool, np.ndarray],
    configclass=OutlineConfig,
    fname='curvrank',
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

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> success_list = ibs.depc_image.get('outline', gid_list, 'success', config=DEFAULT_TEST_CONFIG)
        >>> outlines = ibs.depc_image.get('outline', gid_list, 'outline', config=DEFAULT_TEST_CONFIG)
        >>> outline = outlines[0]
        >>> assert ut.hash_data(outline) in ['qiideplhbdrbvnkkihqeibedbphqzmyw']
        >>> assert success_list == [True]
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
        'scale': config['curvrank_scale'],
        'allow_diagonal': config['outline_allow_diagonal']
    }
    success_list, outlines = ibs.ibeis_plugin_curvrank_outline(*args, **kwargs)
    for success, outline in zip(success_list, outlines):
        yield (
            success,
            outline
        )


class TrailingEdgeConfig(dtool.Config):
    def get_param_info_list(self):
        return []


@register_preproc_image(
    tablename='trailing_edge', parents=['outline'],
    colnames=['success', 'trailing_edge'],
    coltypes=[bool, np.ndarray],
    configclass=TrailingEdgeConfig,
    fname='curvrank',
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

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> success_list = ibs.depc_image.get('trailing_edge', gid_list, 'success', config=DEFAULT_TEST_CONFIG)
        >>> trailing_edges = ibs.depc_image.get('trailing_edge', gid_list, 'trailing_edge', config=DEFAULT_TEST_CONFIG)
        >>> trailing_edge = trailing_edges[0]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(trailing_edge) in ['hspynmqvrnhjmowostnissyymllnbiop']
    """
    ibs = depc.controller

    success_list = depc.get_native('outline', outline_rowid_list, 'success')
    outlines     = depc.get_native('outline', outline_rowid_list, 'outline')

    values = ibs.ibeis_plugin_curvrank_trailing_edges(success_list, outlines)
    success_list, trailing_edges = values

    for success, trailing_edge in zip(success_list, trailing_edges):
        yield (
            success,
            trailing_edge,
        )


class CurvatuveConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('curvature_scales',        DEFAULT_SCALES),
            ut.ParamInfo('curvatute_transpose_dims', False),
        ]


@register_preproc_image(
    tablename='curvature', parents=['trailing_edge'],
    colnames=['success', 'curvature'],
    coltypes=[bool, np.ndarray],
    configclass=CurvatuveConfig,
    fname='curvrank',
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

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> success_list = ibs.depc_image.get('curvature', gid_list, 'success', config=DEFAULT_TEST_CONFIG)
        >>> curvatures = ibs.depc_image.get('curvature', gid_list, 'curvature', config=DEFAULT_TEST_CONFIG)
        >>> curvature = curvatures[0]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(curvature) in ['prakvzmuaeajjcbxjstkydqtdfqxlmpi']
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
            ut.ParamInfo('curvature_scales',                   DEFAULT_SCALES),
            ut.ParamInfo('curvature_descriptor_curv_length',   1024),
            ut.ParamInfo('curvature_descriptor_num_keypoints', 32),
            ut.ParamInfo('curvature_descriptor_uniform',       False),
            ut.ParamInfo('curvature_descriptor_feat_dim',      32),
        ]


@register_preproc_image(
    tablename='curvature_descriptor', parents=['curvature'],
    colnames=['success', 'descriptor'],
    coltypes=[bool, ('extern', ut.load_cPkl, ut.save_cPkl)],
    configclass=CurvatuveDescriptorConfig,
    fname='curvrank',
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

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> success_list = ibs.depc_image.get('curvature_descriptor', gid_list, 'success', config=DEFAULT_TEST_CONFIG)
        >>> curvature_descriptor_dicts = ibs.depc_image.get('curvature_descriptor', gid_list, 'descriptor', config=DEFAULT_TEST_CONFIG)
        >>> curvature_descriptor_dict = curvature_descriptor_dicts[0]
        >>> assert success_list == [True]
        >>> hash_list = [
        >>>     ut.hash_data(curvature_descriptor_dict[scale])
        >>>     for scale in sorted(list(curvature_descriptor_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['pqbzoibzzcndfwemlmudneawivzacupf']
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


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis_curvrank._plugin_depc --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
