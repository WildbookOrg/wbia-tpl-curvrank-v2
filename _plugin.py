from __future__ import absolute_import, division, print_function
from ibeis.control import controller_inject  # NOQA
import utool as ut


# We want to register the depc plugin functions as well, so import it here for IBEIS
import ibeis_curvrank._plugin_depc  # NOQA


_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)
register_api = controller_inject.get_ibeis_flask_api(__name__)


URL_DICT = {
    'localization': 'https://lev.cs.rpi.edu/public/models/curvrank.localization.weights.pkl',
    'segmentation': 'https://lev.cs.rpi.edu/public/models/curvrank.segmentation.weights.pkl',
}


def _assert_hashes(data, hash_list, tag='data'):
    data_hash = ut.hash_data(data)
    print('ut.hash_data(%s) = %r' % (tag, data_hash, ))
    assert data_hash in hash_list


@register_ibs_method
def ibeis_plugin_curvrank_example(ibs):
    from ibeis_curvrank.example_workflow_ibeis import example
    example()


@register_ibs_method
def ibeis_plugin_curvrank_preprocessing(ibs, gid_list, width=256, height=256):
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
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(gid_list)
        >>> resized_images, resized_masks, pre_transforms = values
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
                                       width=256, height=256,
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
        >>> import numpy as np
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> gid_list = ibs.get_valid_gids()[0:1]
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(gid_list)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks)
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

    return localized_images, localized_masks, loc_transforms


@register_ibs_method
def ibeis_plugin_curvrank_refinement(ibs, gid_list, pre_transforms,
                                     loc_transforms, width=256, height=256,
                                     scale=4):
    r"""
    Refine localizations for CurvRank

    Args:
        ibs       (IBEISController): IBEIS controller object
        gid_list  (list of int): list of image rowids (gids)
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
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(gid_list)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks)
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(gid_list, pre_transforms, loc_transforms)
        >>> refined_localizations, refined_masks = values
        >>> refined_localization = refined_localizations[0]
        >>> refined_mask         = refined_masks[0]
        >>> #TODO verify that mac/ubuntu values are consistent on those OSes
        >>> assert ut.hash_data(refined_localization) in ['idspzbmvqxvgoyyjkuseeztpmjkbisrz']
        >>> assert ut.hash_data(refined_mask)         in ['luqzalptfdneljbkslrpufypwmajsmdv']
    """
    import ibeis_curvrank.functional as F

    image_list = ibs.get_images(gid_list)

    metadata_list = ibs.get_image_metadata(gid_list)
    viewpoint_list = [metadata.get('viewpoint', None) for metadata in metadata_list]
    flip_list = [viewpoint == 'right' for viewpoint in viewpoint_list]

    refined_localizations, refined_masks = [], []
    zipped = zip(image_list, flip_list, pre_transforms, loc_transforms)
    for image, flip, pre_transform, loc_transform in zipped:
        refined_localization, refined_mask = F.refine_localization(
            image, flip, pre_transform, loc_transform,
            scale, height, width
        )
        refined_localizations.append(refined_localization)
        refined_masks.append(refined_mask)

    return refined_localizations, refined_masks


@register_ibs_method
def ibeis_plugin_curvrank_segmentation(ibs, refined_localizations, refined_masks,
                                       width=256, height=256, scale=4,
                                       model_tag='segmentation'):
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
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(gid_list)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks)
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(gid_list, pre_transforms, loc_transforms)
        >>> refined_localizations, refined_masks = values
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(refined_localizations, refined_masks)
        >>> segmentations, refined_segmentations = values
        >>> segmentation = segmentations[0]
        >>> refined_segmentation = refined_segmentations[0]
        >>> assert ut.hash_data(segmentation)         in ['wnfimwthormmytbumjnqrhjbsfjccksy']
        >>> assert ut.hash_data(refined_segmentation) in ['fmmuefyrgmpyaaeakqnbgbafrhwbvohf']
    """
    import ibeis_curvrank.functional as F
    from ibeis_curvrank import segmentation, model, theano_funcs

    model_url = URL_DICT.get(model_tag, None)
    assert model_url is not None
    weight_filepath = ut.grab_file_url(model_url, appname='ibeis_curvrank', check_hash=True)

    segmentation_layers = segmentation.build_model_batchnorm_full((None, 3, height, width))

    # I am not sure these are the correct args to load_weights
    model.load_weights(segmentation_layers['seg_out'], weight_filepath)
    segmentation_func = theano_funcs.create_segmentation_func(segmentation_layers)
    values = F.segment_contour(refined_localizations, refined_masks, scale,
                               height, width, segmentation_func)
    segmentations, refined_segmentations = values
    return segmentations, refined_segmentations


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis_curvrank._plugin --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
