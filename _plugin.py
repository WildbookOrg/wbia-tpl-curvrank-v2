from __future__ import absolute_import, division, print_function
from ibeis.control import controller_inject  # NOQA
import numpy as np
import utool as ut
import vtool as vt
import dtool


_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)
register_api = controller_inject.get_ibeis_flask_api(__name__)
register_preproc_image = controller_inject.register_preprocs['image']
register_preproc_annot = controller_inject.register_preprocs['annot']


URL_DICT = {
    'localization': 'https://lev.cs.rpi.edu/public/models/curvrank.localization.weights.pkl',
    'segmentation': 'https://lev.cs.rpi.edu/public/models/curvrank.segmentation.weights.pkl',
}


# def _convert_np_type(image_list, new_np_type=np.uint8, around=True):
#     image_list_ = []
#     for image in image_list:
#         if around:
#             image = np.around(image)
#         image = image.astype(new_np_type)
#         image_list_.append(image)

#     return image_list_


def _assert_hashes(data, hash_list, tag='data'):
    data_hash = ut.hash_data(data)
    print('ut.hash_data(%s) = %r' % (tag, data_hash, ))
    assert data_hash in hash_list


@register_ibs_method
def ibeis_plugin_curvrank_example(ibs):
    from ibeis_curvrank.example_workflow import example
    example()


class PreprocessConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('preprocess_height', 256),
            ut.ParamInfo('preprocess_width', 256),
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
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_preprocessing_depc

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> config = {
        >>>     'preprocess_height': 256,
        >>>     'preprocess_width': 256,
        >>> }
        >>> resized_images = ibs.depc_image.get('preprocess', gid_list, 'resized_img',  config=config)
        >>> resized_masks  = ibs.depc_image.get('preprocess', gid_list, 'mask_img',     config=config)
        >>> pre_transforms = ibs.depc_image.get('preprocess', gid_list, 'pretransform', config=config)
        >>> resized_image = resized_images[0]
        >>> resized_mask  = resized_masks[0]
        >>> pre_transform = pre_transforms[0]
        >>> print('ut.hash_data(resized_image) = %r' % (ut.hash_data(resized_image), ))
        >>> print('ut.hash_data(resized_mask)  = %r' % (ut.hash_data(resized_mask), ))
        >>> assert ut.hash_data(resized_image) in ['ynbsgqgfutslspmatpenvcbtgedsyzoo']
        >>> assert ut.hash_data(resized_mask)  in ['mnhartnytowmmhskblocubqmzhbofynr']
        >>> result = pre_transform
        >>> print(result)
        [[ 0.11077456  0.          0.        ]
         [ 0.          0.11077456 38.        ]
         [ 0.          0.          1.        ]]
    """
    height = config['preprocess_height']
    width = config['preprocess_width']

    ibs = depc.controller
    values = ibs.ibeis_plugin_curvrank_preprocessing(gid_list, height=height, width=width)
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


@register_ibs_method
def ibeis_plugin_curvrank_preprocessing(ibs, gid_list, height=256, width=256):
    r"""
    Pre-process images for CurvRank

    Args:
        ibs       (IBEISController): IBEIS controller object
        gid_list  (list of int): list of image rowids (gids)

    Returns:
        resized_images
        resized_masks
        pre_transforms

    CommandLine:
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_preprocessing

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(gid_list, width=256, height=256)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> resized_image = resized_images[0]
        >>> resized_mask  = resized_masks[0]
        >>> pre_transform = pre_transforms[0]
        >>> print('ut.hash_data(resized_image) = %r' % (ut.hash_data(resized_image), ))
        >>> print('ut.hash_data(resized_mask)  = %r' % (ut.hash_data(resized_mask), ))
        >>> assert ut.hash_data(resized_image) in ['ynbsgqgfutslspmatpenvcbtgedsyzoo']
        >>> assert ut.hash_data(resized_mask)  in ['mnhartnytowmmhskblocubqmzhbofynr']
        >>> result = pre_transform
        >>> print(result)
        [[ 0.11077456  0.          0.        ]
         [ 0.          0.11077456 38.        ]
         [ 0.          0.          1.        ]]
    """
    import ibeis_curvrank.functional as F

    image_list = ibs.get_images(gid_list)

    metadata_list = ibs.get_image_metadata(gid_list)
    viewpoint_list = [metadata.get('viewpoint', None) for metadata in metadata_list]
    flip_list = [viewpoint == 'right' for viewpoint in viewpoint_list]

    resized_images, resized_masks, pre_transforms = [], [], []
    for image, flip in zip(image_list, flip_list):
        vals = F.preprocess_image(image, flip, height, width)

        resized_image, resized_mask, pre_transform = vals
        resized_images.append(resized_image)
        resized_masks.append(resized_mask)
        pre_transforms.append(pre_transform)

    return resized_images, resized_masks, pre_transforms


@register_ibs_method
def ibeis_plugin_curvrank_localization(ibs, resized_images, resized_masks,
                                       model_tag='localization'):
    r"""
    Localize images for CurvRank

    Args:
        ibs       (IBEISController): IBEIS controller object
        model_tag  (string): Key to URL_DICT entry for this model
        resized_images (list of np.ndarray): widthXheightX3 color channels
        resized_masks (list of np.ndarray): heightXwidth greyscale images

    Returns:
        localized_images
        localized_masks
        loc_transforms

    CommandLine:
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_localization

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(gid_list, width=256, height=256)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images,resized_masks)
        >>> localized_images, localized_masks, loc_transforms = values
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
    import ibeis_curvrank.functional as F
    from ibeis_curvrank import localization, model, theano_funcs

    model_url = URL_DICT.get(model_tag, None)
    assert model_url is not None
    weight_filepath = ut.grab_file_url(model_url, appname='ibeis_curvrank', check_hash=True)

    # Make sure resized images all have the same shape
    height, width = resized_images[0].shape[:2]
    assert np.all([
        resized_image.shape[0] == height and resized_image.shape[1] == width
        for resized_image in resized_images
    ])

    layers = localization.build_model((None, 3, height, width))
    model.load_weights(
        [
            layers['trans'],
            layers['loc']
        ],
        weight_filepath
    )
    localization_func = theano_funcs.create_localization_infer_func(layers)
    values = F.localize(resized_images, resized_masks, height, width, localization_func)
    localized_images, localized_masks, loc_transforms = values

    assert np.all([
        localized_image.shape == resized_image.shape
        for localized_image, resized_image in zip(localized_images, resized_images)
    ])

    return localized_images, localized_masks, loc_transforms


class LocalizationConfig(dtool.Config):
    def get_param_info_list(self):
        return [
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
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_localization_depc

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> config = {
        >>>     'preprocess_height': 256,
        >>>     'preprocess_width': 256,
        >>>     'localization_model_tag': 'localization'
        >>> }
        >>> localized_images = ibs.depc_image.get('localization', gid_list, 'localized_img',  config=config)
        >>> localized_masks  = ibs.depc_image.get('localization', gid_list, 'mask_img',     config=config)
        >>> loc_transforms = ibs.depc_image.get('localization', gid_list, 'transform', config=config)
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
    model_tag = config['localization_model_tag']
    ibs = depc.controller
    # fetch resized image
    resized_images = depc.get_native('preprocess', preprocess_rowid_list, 'resized_img')
    # fetch resized mask
    resized_masks = depc.get_native('preprocess', preprocess_rowid_list, 'mask_img')
    # call function above
    height, width = resized_images[0].shape[:2]

    values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks,
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


@register_ibs_method
def ibeis_plugin_curvrank_refinement(ibs, gid_list, localized_images,
                                     pre_transforms, loc_transforms, scale=4):
    r"""
    Refine localizations for CurvRank

    Args:
        ibs       (IBEISController): IBEIS controller object
        gid_list  (list of int): list of image rowids (gids)
        localized_images  (list of np.ndarray): widthXheightX3 color channels
        pre_transforms (list of np.ndarray):
        loc_transforms (list of np.ndarray):
        scale (int): how many scales to perform the refinement (default to 4).

    Returns:
        refined_localizations
        refined_masks

    CommandLine:
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_refinement

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(gid_list, width=256, height=256)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images,resized_masks)
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(gid_list, localized_images,
        >>>                                               pre_transforms, loc_transforms, scale=4)
        >>> refined_localizations, refined_masks = values
        >>> refined_localization = refined_localizations[0]
        >>> refined_mask         = refined_masks[0]
        >>> #TODO verify that mac/ubuntu values are consistent on those OSes
        >>> assert ut.hash_data(refined_localization) in ['hslglhazpolotapwmpjyymjprtidgusb', 'fwjorhantaihpnlptakncuwrbivsnogr']
        >>> assert ut.hash_data(refined_mask)         in ['addlxdyjkminxlfsdfqmmuptyprhpyfi']
    """
    import ibeis_curvrank.functional as F

    metadata_list = ibs.get_image_metadata(gid_list)
    viewpoint_list = [metadata.get('viewpoint', None) for metadata in metadata_list]
    flip_list = [viewpoint == 'right' for viewpoint in viewpoint_list]

    ut.embed()

    refined_localizations, refined_masks = [], []
    zipped = zip(localized_images, flip_list, pre_transforms, loc_transforms)
    for localized_image, flip, pre_transform, loc_transform in zipped:
        height, width = localized_image.shape[:2]
        refined_localization, refined_mask = F.refine_localization(
            localized_image, flip, pre_transform, loc_transform,
            scale, height, width
        )
        refined_localizations.append(refined_localization)
        refined_masks.append(refined_mask)

    return refined_localizations, refined_masks


class RefinementConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('curvrank_scale', 4),
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
def ibeis_plugin_curvrank_refinement_depc(depc, localization_rowid_list, preprocess_rowid_list, config=None):
    r"""
    Refine localizations for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_refinement_depc

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> config = {
        >>>     'preprocess_height': 256,
        >>>     'preprocess_width': 256,
        >>>     'localization_model_tag': 'localization',
        >>>     'curvrank_scale': 4
        >>> }
        >>> refined_localizations = ibs.depc_image.get('refinement', gid_list, 'refined_img', config=config)
        >>> refined_masks         = ibs.depc_image.get('refinement', gid_list, 'mask_img', config=config)
        >>> refined_localization  = refined_localizations[0]
        >>> refined_mask          = refined_masks[0]
        >>> #TODO verify that mac/ubuntu values are consistent on those OSes
        >>> assert ut.hash_data(refined_localization) in ['fwjorhantaihpnlptakncuwrbivsnogr', 'hslglhazpolotapwmpjyymjprtidgusb']
        >>> assert ut.hash_data(refined_mask)         in ['addlxdyjkminxlfsdfqmmuptyprhpyfi']
    """
    ibs = depc.controller
    scale = config['curvrank_scale']

    localized_images = depc.get_native('localization', localization_rowid_list, 'localized_img')
    loc_transforms   = depc.get_native('localization', localization_rowid_list, 'transform')
    pre_transforms   = depc.get_native('preprocess',   preprocess_rowid_list,   'pretransform')

    values = ibs.ibeis_plugin_curvrank_refinement(preprocess_rowid_list, localized_images,
                                                  pre_transforms, loc_transforms, scale)
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


@register_ibs_method
def ibeis_plugin_curvrank_segmentation(ibs, refined_localizations, refined_masks,
                                       scale=4, model_tag='segmentation'):
    r"""
    Localize images for CurvRank

    Args:
        ibs       (IBEISController): IBEIS controller object
        refined_localizations: output of ibeis_plugin_curvrank_refinement
        refined_masks: output of ibeis_plugin_curvrank_refinement
        model_tag  (string): Key to URL_DICT entry for this model
        scale (int): how many scales to perform the refinement (default to 4).

    Returns:
        segmentations
        refined_segmentations

    CommandLine:
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_segmentation

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(gid_list, width=256, height=256)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images,resized_masks)
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(gid_list, localized_images,
        >>>                                               pre_transforms, loc_transforms, scale=4)
        >>> refined_localizations, refined_masks = values
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(refined_localizations, refined_masks, scale=4)
        >>> segmentations, refined_segmentations = values
        >>> segmentation = segmentations[0]
        >>> refined_segmentation = refined_segmentations[0]
        >>> assert ut.hash_data(segmentation)         in ['pislgcxekvrzeabsyfbksycesellsldw', 'ciruuvnvemwjfmfoermdvixdenkfiwbl']
        >>> assert ut.hash_data(refined_segmentation) in ['fropddwbykfltfjdqqsjvqoadmqwxszk', 'whbqxdumhmtzvxprfsqkhtdqimzxcdui']
    """
    import ibeis_curvrank.functional as F
    from ibeis_curvrank import segmentation, model, theano_funcs

    model_url = URL_DICT.get(model_tag, None)
    assert model_url is not None
    weight_filepath = ut.grab_file_url(model_url, appname='ibeis_curvrank', check_hash=True)

    # derive height and width from scale and images
    height, width = refined_localizations[0].shape[:2]
    height = height // scale
    width  = width  // scale

    segmentation_layers = segmentation.build_model_batchnorm_full((None, 3, height, width))

    # I am not sure these are the correct args to load_weights
    model.load_weights(segmentation_layers['seg_out'], weight_filepath)
    segmentation_func = theano_funcs.create_segmentation_func(segmentation_layers)
    values = F.segment_contour(refined_localizations, refined_masks, scale,
                               height, width, segmentation_func)
    segmentations, refined_segmentations = values
    return segmentations, refined_segmentations


class SegmentationConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('segmentation_model_tag', 'segmentation'),
            ut.ParamInfo('curvrank_scale', 4),
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
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_segmentation_depc

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> config = {
        >>>     'preprocess_height': 256,
        >>>     'preprocess_width': 256,
        >>>     'localization_model_tag': 'localization',
        >>>     'curvrank_scale': 4
        >>> }
        >>> segmentations          = ibs.depc_image.get('segmentation', gid_list, 'segmentations_img', config=config)
        >>> refined_segmentations  = ibs.depc_image.get('segmentation', gid_list, 'refined_segmentations_img', config=config)
        >>> segmentation           = segmentations[0]
        >>> refined_segmentation   = refined_segmentations[0]
        >>> assert ut.hash_data(segmentation)         in ['pislgcxekvrzeabsyfbksycesellsldw', 'ciruuvnvemwjfmfoermdvixdenkfiwbl']
        >>> assert ut.hash_data(refined_segmentation) in ['fropddwbykfltfjdqqsjvqoadmqwxszk', 'whbqxdumhmtzvxprfsqkhtdqimzxcdui']
    """
    ibs = depc.controller

    print(config)

    scale = config['curvrank_scale']
    model_tag = config['segmentation_model_tag']

    refined_localizations = depc.get_native('refinement', refinement_rowid_list, 'refined_img')
    refined_masks         = depc.get_native('refinement', refinement_rowid_list, 'mask_img')

    values = ibs.ibeis_plugin_curvrank_segmentation(refined_localizations, refined_masks,
                                                    scale=scale, model_tag=model_tag)
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


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis_curvrank._plugin
        python -m ibeis_curvrank._plugin --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
