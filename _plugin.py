from __future__ import absolute_import, division, print_function
from ibeis.control import controller_inject  # NOQA
from os.path import abspath, join, exists
import ibeis_curvrank.functional as F
import numpy as np
import utool as ut


# We want to register the depc plugin functions as well, so import it here for IBEIS
import ibeis_curvrank._plugin_depc  # NOQA
from ibeis_curvrank._plugin_depc import DEFAULT_SCALES


_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)
register_api = controller_inject.get_ibeis_flask_api(__name__)


USE_DEPC = True


URL_DICT = {
    'dorsal': {
        'localization': 'https://lev.cs.rpi.edu/public/models/curvrank.localization.dorsal.weights.pkl',
        'segmentation': 'https://lev.cs.rpi.edu/public/models/curvrank.segmentation.dorsal.weights.pkl',
    },
    'fluke': {
        'localization': None,
        'segmentation': 'https://lev.cs.rpi.edu/public/models/curvrank.segmentation.fluke.weights.pkl',
    },
}


@register_ibs_method
def ibeis_plugin_curvrank_preprocessing(ibs, aid_list, width=256, height=256, **kwargs):
    r"""
    Pre-process images for CurvRank

    Args:
        ibs       (IBEISController): IBEIS controller object
        aid_list  (list of int): list of image rowids (aids)

    Returns:
        resized_images
        resized_masks
        pre_transforms

    CommandLine:
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_preprocessing
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_preprocessing:0
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_preprocessing:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list)
        >>> resized_images, resized_masks, pre_transforms = values
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

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> model_type = 'fluke'
        >>> width = DEFAULT_WIDTH[model_type]
        >>> height = DEFAULT_HEIGHT[model_type]
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list, width=width, height=height)
        >>> resized_images, resized_masks, pre_transforms = values
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
    # ibs._parallel_chips = True
    image_list = ibs.get_annot_chips(aid_list)

    viewpoint_list = ibs.get_annot_viewpoints(aid_list)
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
                                       width=256, height=256, model_type='dorsal',
                                       model_tag='localization', **kwargs):
    r"""
    Localize images for CurvRank

    Args:
        ibs       (IBEISController): IBEIS controller object
        model_tag  (string): Key to URL_DICT entry for this model
        resized_images (list of np.ndarray): (height, width, 3) BGR images
        resized_masks (list of np.ndarray): (height, width) binary masks

    Returns:
        localized_images
        localized_masks
        loc_transforms

    CommandLine:
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_localization
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_localization:0
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_localization:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks)
        >>> localized_images, localized_masks, loc_transforms = values
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
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> model_type = 'fluke'
        >>> width = DEFAULT_WIDTH[model_type]
        >>> height = DEFAULT_HEIGHT[model_type]
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list, width=width, height=height)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks, width=width, height=height, model_type=model_type)
        >>> localized_images, localized_masks, loc_transforms = values
        >>> loc_transform = loc_transforms[0]
        >>> # localized_image appears to differ very slightly in ubuntu vs. mac. Hashes below for each respectively.
        >>> #TODO verify that mac/ubuntu values are consistent on those OSes
        >>> assert ut.hash_data(resized_images) == ut.hash_data(localized_images)
        >>> assert ut.hash_data(resized_masks)  == ut.hash_data(localized_masks)
        >>> assert np.sum(loc_transform) == 3.0
    """
    from ibeis_curvrank import localization, model, theano_funcs

    model_url = URL_DICT.get(model_type, {}).get(model_tag, None)

    if model_url is None:
        localized_images = resized_images
        localized_masks = resized_masks
        loc_transforms = [np.eye(3, dtype=np.float32)] * len(localized_images)
    else:
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
def ibeis_plugin_curvrank_refinement(ibs, aid_list, pre_transforms,
                                     loc_transforms, width=256, height=256,
                                     scale=4, **kwargs):
    r"""
    Refine localizations for CurvRank

    Args:
        ibs       (IBEISController): IBEIS controller object
        aid_list  (list of int): list of image rowids (aids)
        pre_transforms (list of np.ndarray): (3, 3) similarity matrices
        loc_transforms (list of np.ndarray): (3, 3) affine matrices
        scale (int): upsampling factor from coarse to fine-grained (default to 4).

    Returns:
        refined_localizations
        refined_masks

    CommandLine:
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_refinement
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_refinement:0
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_refinement:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks)
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms)
        >>> refined_localizations, refined_masks = values
        >>> refined_localization = refined_localizations[0]
        >>> refined_mask         = refined_masks[0]
        >>> #TODO verify that mac/ubuntu values are consistent on those OSes
        >>> assert ut.hash_data(refined_localization) in ['nxhumkmybgbjdjcffuneozzmptvivvlh']
        >>> assert ut.hash_data(refined_mask)         in ['bwuzcdgbfyqhzgdthazfgegbzeykvbnt']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> model_type = 'fluke'
        >>> width = DEFAULT_WIDTH[model_type]
        >>> height = DEFAULT_HEIGHT[model_type]
        >>> scale = DEFAULT_SCALE[model_type]
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list, width=width, height=height)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks, width=width, height=height, model_type=model_type)
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms, width=width, height=height, scale=scale)
        >>> refined_localizations, refined_masks = values
        >>> refined_localization = refined_localizations[0]
        >>> refined_mask         = refined_masks[0]
        >>> #TODO verify that mac/ubuntu values are consistent on those OSes
        >>> assert ut.hash_data(refined_localization) in ['cwmqsvpabxdaftsnupgerivjufsavfhl']
        >>> assert ut.hash_data(refined_mask)         in ['zwfgmumqblkfzejnseauggiedzpbbjoa']
    """
    # ibs._parallel_chips = True
    image_list = ibs.get_annot_chips(aid_list)

    viewpoint_list = ibs.get_annot_viewpoints(aid_list)
    flip_list = [viewpoint == 'right' for viewpoint in viewpoint_list]

    OLD = True

    if OLD:
        refined_localizations, refined_masks = [], []
        zipped = zip(image_list, flip_list, pre_transforms, loc_transforms)
        for image, flip, pre_transform, loc_transform in zipped:
            refined_localization, refined_mask = F.refine_localization(
                image, flip, pre_transform, loc_transform,
                scale, height, width
            )
            refined_localizations.append(refined_localization)
            refined_masks.append(refined_mask)

    else:
        scale_list  = [scale]  * len(aid_list)
        height_list = [height] * len(aid_list)
        width_list  = [width]  * len(aid_list)

        zipped = zip(image_list, flip_list, pre_transforms, loc_transforms,
                     scale_list, height_list, width_list)

        config_ = {
            'ordered': True,
            'chunksize': 32,
            'force_serial': ibs.force_serial,
        }
        generator = ut.generate2(F.refine_localization, zipped, nTasks=len(aid_list), **config_)

        refined_localizations, refined_masks = [], []
        for refined_localization, refined_mask in generator:
            refined_localizations.append(refined_localization)
            refined_masks.append(refined_mask)

    return refined_localizations, refined_masks


@register_ibs_method
def ibeis_plugin_curvrank_segmentation(ibs, refined_localizations, refined_masks,
                                       width=256, height=256, scale=4, model_type='dorsal',
                                       model_tag='segmentation', **kwargs):
    r"""
    Localize images for CurvRank

    Args:
        ibs       (IBEISController): IBEIS controller object
        refined_localizations: output of ibeis_plugin_curvrank_refinement
        refined_masks: output of ibeis_plugin_curvrank_refinement
        model_tag  (string): Key to URL_DICT entry for this model
        scale (int): upsampling factor from coarse to fine-grained (default to 4).

    Returns:
        segmentations
        refined_segmentations

    CommandLine:
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_segmentation
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_segmentation:0
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_segmentation:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks)
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms)
        >>> refined_localizations, refined_masks = values
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(refined_localizations, refined_masks)
        >>> segmentations, refined_segmentations = values
        >>> segmentation = segmentations[0]
        >>> refined_segmentation = refined_segmentations[0]
        >>> assert ut.hash_data(segmentation)         in ['tcfybjuqszadvmfetzxivcvihfkudvqh']
        >>> assert ut.hash_data(refined_segmentation) in ['snjswkyqprmhmpefiiiapdsytubfvcwo']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> model_type = 'fluke'
        >>> width = DEFAULT_WIDTH[model_type]
        >>> height = DEFAULT_HEIGHT[model_type]
        >>> scale = DEFAULT_SCALE[model_type]
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list, width=width, height=height)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks, width=width, height=height, model_type=model_type)
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms, width=width, height=height, scale=scale)
        >>> refined_localizations, refined_masks = values
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(refined_localizations, refined_masks, width=width, height=height, scale=scale, model_type=model_type)
        >>> segmentations, refined_segmentations = values
        >>> segmentation = segmentations[0]
        >>> refined_segmentation = refined_segmentations[0]
        >>> assert ut.hash_data(segmentation)         in ['htbsspdnjfchswtcboifeybpkhmbdmms']
        >>> assert ut.hash_data(refined_segmentation) in ['hqngsbdctbjsruuwjhhbuamcbukbyaea']
    """
    from ibeis_curvrank import segmentation, model, theano_funcs

    model_url = URL_DICT.get(model_type, {}).get(model_tag, None)
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


@register_ibs_method
def ibeis_plugin_curvrank_keypoints(ibs, segmentations, localized_masks,
                                    model_type='dorsal', **kwargs):
    r"""
    Args:
        ibs       (IBEISController): IBEIS controller object
        segmentations: output of ibeis_plugin_curvrank_segmentation
        refined_masks: output of ibeis_plugin_curvrank_refinement

    Returns:
        success_list: bool list
        starts: list of keypoint starts
        ends: list of keypoint ends

    CommandLine:
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_keypoints
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_keypoints:0
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_keypoints:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks)
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms)
        >>> refined_localizations, refined_masks = values
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(refined_localizations, refined_masks)
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks)
        >>> success_list, starts, ends = values
        >>> start = tuple(starts[0])
        >>> end = tuple(ends[0])
        >>> assert success_list == [True]
        >>> assert start == (203, 3)
        >>> assert end   == (198, 252)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> model_type = 'fluke'
        >>> width = DEFAULT_WIDTH[model_type]
        >>> height = DEFAULT_HEIGHT[model_type]
        >>> scale = DEFAULT_SCALE[model_type]
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list, width=width, height=height)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks, width=width, height=height, model_type=model_type)
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms, width=width, height=height, scale=scale)
        >>> refined_localizations, refined_masks = values
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(refined_localizations, refined_masks, width=width, height=height, scale=scale, model_type=model_type)
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks, model_type=model_type)
        >>> success_list, starts, ends = values
        >>> start = tuple(starts[0])
        >>> end = tuple(ends[0])
        >>> assert success_list == [True]
        >>> assert start == (56, 8)
        >>> assert end   == (59, 358)
    """
    if model_type == 'dorsal':
        from ibeis_curvrank.dorsal_utils import find_dorsal_keypoints as find_func
    else:
        from ibeis_curvrank.dorsal_utils import find_fluke_keypoints as find_func

    starts, ends, success_list = [], [], []

    for segmentation, localized_mask in zip(segmentations, localized_masks):
        start, end = F.find_keypoints(
            find_func,
            segmentation,
            localized_mask
        )

        success = start is not None and end is not None

        success_list.append(success)
        starts.append(start)
        ends.append(end)
    return success_list, starts, ends


@register_ibs_method
def ibeis_plugin_curvrank_outline(ibs, success_list, starts, ends,
                                  refined_localizations, refined_masks,
                                  refined_segmentations, scale=4,
                                  model_type='dorsal', allow_diagonal=False,
                                  **kwargs):
    r"""
    Args:
        ibs       (IBEISController): IBEIS controller object
        success_list: output of ibeis_plugin_curvrank_keypoints
        starts: output of ibeis_plugin_curvrank_keypoints
        ends: output of ibeis_plugin_curvrank_keypoints
        refined_localizations: output of ibeis_plugin_curvrank_refinement
        refined_masks: output of ibeis_plugin_curvrank_refinement
        refined_segmentations: output of ibeis_plugin_curvrank_refinement
    Returns:
        success_list
        outlines

    CommandLine:
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_outline
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_outline:0
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_outline:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks)
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms)
        >>> refined_localizations, refined_masks = values
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(refined_localizations, refined_masks)
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks)
        >>> success_list, starts, ends = values
        >>> args = success_list, starts, ends, refined_localizations, refined_masks, refined_segmentations
        >>> success_list, outlines = ibs.ibeis_plugin_curvrank_outline(*args)
        >>> outline = outlines[0]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(outline) in ['lyrkwgzncvjpjvovikkvspdkecardwyz']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> model_type = 'fluke'
        >>> width = DEFAULT_WIDTH[model_type]
        >>> height = DEFAULT_HEIGHT[model_type]
        >>> scale = DEFAULT_SCALE[model_type]
        >>> allow_diagonal = DEFAULT_ALLOW_DIAGONAL[model_type]
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list, width=width, height=height)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks, width=width, height=height, model_type=model_type)
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms, width=width, height=height, scale=scale)
        >>> refined_localizations, refined_masks = values
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(refined_localizations, refined_masks, width=width, height=height, scale=scale, model_type=model_type)
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks, model_type=model_type)
        >>> success_list, starts, ends = values
        >>> args = success_list, starts, ends, refined_localizations, refined_masks, refined_segmentations
        >>> success_list, outlines = ibs.ibeis_plugin_curvrank_outline(*args, scale=scale, model_type=model_type, allow_diagonal=allow_diagonal)
        >>> outline = outlines[0]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(outline) in ['qqvetxfhhipfuqneuinwrvcztkjlfoak']
    """
    if model_type == 'dorsal':
        from ibeis_curvrank.dorsal_utils import dorsal_cost_func as cost_func
    else:
        from ibeis_curvrank.dorsal_utils import fluke_cost_func as cost_func

    success_list_ = []
    outlines = []
    zipped = zip(success_list, starts, ends, refined_localizations,
                 refined_masks, refined_segmentations)
    for value in zipped:
        success, start, end, refined_loc, refined_mask, refined_seg = value
        success_ = success
        if success:
            start = np.array(start, dtype=np.int32)
            end   = np.array(end,   dtype=np.int32)
            outline = F.extract_outline(
                refined_loc, refined_mask, refined_seg, scale, start, end,
                cost_func, allow_diagonal)
            if outline is None:
                success_ = False
        else:
            outline = None

        success_list_.append(success_)
        outlines.append(outline)

    return success_list_, outlines


@register_ibs_method
def ibeis_plugin_curvrank_trailing_edges(ibs, success_list, outlines, model_type='dorsal',
                                         **kwargs):
    r"""
    Args:
        ibs       (IBEISController): IBEIS controller object
        success_list: output of ibeis_plugin_curvrank_outline
        outlines (list of np.ndarray): output of ibeis_plugin_curvrank_outline

    Returns:
        success_list_
        trailing_edges

    CommandLine:
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_trailing_edges
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_trailing_edges:0
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_trailing_edges:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks)
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms)
        >>> refined_localizations, refined_masks = values
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(refined_localizations, refined_masks)
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks)
        >>> success_list, starts, ends = values
        >>> args = success_list, starts, ends, refined_localizations, refined_masks, refined_segmentations
        >>> success_list, outlines = ibs.ibeis_plugin_curvrank_outline(*args)
        >>> values = ibs.ibeis_plugin_curvrank_trailing_edges(success_list, outlines)
        >>> success_list, trailing_edges = values
        >>> trailing_edge = trailing_edges[0]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(trailing_edge) in ['wiabdtkbaqjuvszkyvyjnpomrivyadaa']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> model_type = 'fluke'
        >>> width = DEFAULT_WIDTH[model_type]
        >>> height = DEFAULT_HEIGHT[model_type]
        >>> scale = DEFAULT_SCALE[model_type]
        >>> allow_diagonal = DEFAULT_ALLOW_DIAGONAL[model_type]
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list, width=width, height=height)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks, width=width, height=height, model_type=model_type)
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms, width=width, height=height, scale=scale)
        >>> refined_localizations, refined_masks = values
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(refined_localizations, refined_masks, width=width, height=height, scale=scale, model_type=model_type)
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks, model_type=model_type)
        >>> success_list, starts, ends = values
        >>> args = success_list, starts, ends, refined_localizations, refined_masks, refined_segmentations
        >>> values = ibs.ibeis_plugin_curvrank_outline(*args, scale=scale, model_type=model_type, allow_diagonal=allow_diagonal)
        >>> success_list, outlines = values
        >>> values = ibs.ibeis_plugin_curvrank_trailing_edges(success_list, outlines, model_type=model_type)
        >>> success_list, trailing_edges = values
        >>> assert success_list == [True]
        >>> assert ut.hash_data(outlines) == ut.hash_data(trailing_edges)
    """
    from ibeis_curvrank.dorsal_utils import separate_leading_trailing_edges

    if model_type == 'dorsal':
        success_list_ = []
        trailing_edges = []
        for success, outline in zip(success_list, outlines):
            success_ = success
            if success:
                values = F.separate_edges(separate_leading_trailing_edges, outline)
                _, trailing_edge = values

                if trailing_edge is None:
                    success_ = False
            else:
                trailing_edge = None

            success_list_.append(success_)
            trailing_edges.append(trailing_edge)
    else:
        success_list_ = success_list
        trailing_edges = outlines

    return success_list_, trailing_edges


@register_ibs_method
def ibeis_plugin_curvrank_curvatures(ibs, success_list, trailing_edges,
                                     scales=DEFAULT_SCALES['dorsal'], transpose_dims=False,
                                     **kwargs):
    r"""
    Args:
        ibs       (IBEISController): IBEIS controller object
        success_list: output of ibeis_plugin_curvrank_outline
        outlines (list of np.ndarray): output of ibeis_plugin_curvrank_outline

    Returns:
        success_list_
        curvatures

    CommandLine:
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_curvatures
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_curvatures:0
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_curvatures:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks)
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms)
        >>> refined_localizations, refined_masks = values
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(refined_localizations, refined_masks)
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks)
        >>> success_list, starts, ends = values
        >>> args = success_list, starts, ends, refined_localizations, refined_masks, refined_segmentations
        >>> success_list, outlines = ibs.ibeis_plugin_curvrank_outline(*args)
        >>> values = ibs.ibeis_plugin_curvrank_trailing_edges(success_list, outlines)
        >>> success_list, trailing_edges = values
        >>> values = ibs.ibeis_plugin_curvrank_curvatures(success_list, trailing_edges)
        >>> success_list, curvatures = values
        >>> curvature = curvatures[0]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(curvature) in ['yeyykrdbfxqyrbdumvpkvatjoddavdgn']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> model_type = 'fluke'
        >>> width = DEFAULT_WIDTH[model_type]
        >>> height = DEFAULT_HEIGHT[model_type]
        >>> scale = DEFAULT_SCALE[model_type]
        >>> scales = DEFAULT_SCALES[model_type]
        >>> allow_diagonal = DEFAULT_ALLOW_DIAGONAL[model_type]
        >>> transpose_dims = DEFAULT_TRANSPOSE_DIMS[model_type]
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list, width=width, height=height)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks, width=width, height=height, model_type=model_type)
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms, width=width, height=height, scale=scale)
        >>> refined_localizations, refined_masks = values
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(refined_localizations, refined_masks, width=width, height=height, scale=scale, model_type=model_type)
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks, model_type=model_type)
        >>> success_list, starts, ends = values
        >>> args = success_list, starts, ends, refined_localizations, refined_masks, refined_segmentations
        >>> values = ibs.ibeis_plugin_curvrank_outline(*args, scale=scale, model_type=model_type, allow_diagonal=allow_diagonal)
        >>> success_list, outlines = values
        >>> values = ibs.ibeis_plugin_curvrank_trailing_edges(success_list, outlines, model_type=model_type)
        >>> success_list, trailing_edges = values
        >>> values = ibs.ibeis_plugin_curvrank_curvatures(success_list, trailing_edges, scales=scales, transpose_dims=transpose_dims)
        >>> success_list, curvatures = values
        >>> curvature = curvatures[0]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(curvature) in ['dpbusvatdgcdblmtwvodvlsnjuffdylp']
    """
    success_list_ = []
    curvatures = []
    for success, trailing_edge in zip(success_list, trailing_edges):
        success_ = success
        if success:
            curvature = F.compute_curvature(trailing_edge, scales, transpose_dims)

            if curvature is None:
                success_ = False
        else:
            curvature = None

        success_list_.append(success_)
        curvatures.append(curvature)

    return success_list_, curvatures


@register_ibs_method
def ibeis_plugin_curvrank_curvature_descriptors(ibs, success_list, curvatures,
                                                curv_length=1024, scales=DEFAULT_SCALES['dorsal'],
                                                num_keypoints=32, uniform=False,
                                                feat_dim=32, **kwargs):
    r"""
    Args:
        ibs       (IBEISController): IBEIS controller object
        success_list: output of ibeis_plugin_curvrank_outline
        outlines (list of np.ndarray): output of ibeis_plugin_curvrank_outline

    Returns:
        success_list_
        curvature_descriptors

    CommandLine:
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_curvature_descriptors
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_curvature_descriptors:0
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_curvature_descriptors:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks)
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms)
        >>> refined_localizations, refined_masks = values
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(refined_localizations, refined_masks)
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks)
        >>> success_list, starts, ends = values
        >>> args = success_list, starts, ends, refined_localizations, refined_masks, refined_segmentations
        >>> success_list, outlines = ibs.ibeis_plugin_curvrank_outline(*args)
        >>> values = ibs.ibeis_plugin_curvrank_trailing_edges(success_list, outlines)
        >>> success_list, trailing_edges = values
        >>> values = ibs.ibeis_plugin_curvrank_curvatures(success_list, trailing_edges)
        >>> success_list, curvatures = values
        >>> values = ibs.ibeis_plugin_curvrank_curvature_descriptors(success_list, curvatures)
        >>> success_list, curvature_descriptor_dicts = values
        >>> curvature_descriptor_dict = curvature_descriptor_dicts[0]
        >>> assert success_list == [True]
        >>> hash_list = [
        >>>     ut.hash_data(curvature_descriptor_dict[scale])
        >>>     for scale in sorted(list(curvature_descriptor_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['mkhgqrrkhisuaenxkuxgbbcqpdfpoofp']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> model_type = 'fluke'
        >>> width = DEFAULT_WIDTH[model_type]
        >>> height = DEFAULT_HEIGHT[model_type]
        >>> scale = DEFAULT_SCALE[model_type]
        >>> scales = DEFAULT_SCALES[model_type]
        >>> allow_diagonal = DEFAULT_ALLOW_DIAGONAL[model_type]
        >>> transpose_dims = DEFAULT_TRANSPOSE_DIMS[model_type]
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list, width=width, height=height)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks, width=width, height=height, model_type=model_type)
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms, width=width, height=height, scale=scale)
        >>> refined_localizations, refined_masks = values
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(refined_localizations, refined_masks, width=width, height=height, scale=scale, model_type=model_type)
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks, model_type=model_type)
        >>> success_list, starts, ends = values
        >>> args = success_list, starts, ends, refined_localizations, refined_masks, refined_segmentations
        >>> values = ibs.ibeis_plugin_curvrank_outline(*args, scale=scale, model_type=model_type, allow_diagonal=allow_diagonal)
        >>> success_list, outlines = values
        >>> values = ibs.ibeis_plugin_curvrank_trailing_edges(success_list, outlines, model_type=model_type)
        >>> success_list, trailing_edges = values
        >>> values = ibs.ibeis_plugin_curvrank_curvatures(success_list, trailing_edges, scales=scales, transpose_dims=transpose_dims)
        >>> success_list, curvatures = values
        >>> values = ibs.ibeis_plugin_curvrank_curvature_descriptors(success_list, curvatures, scales=scales)
        >>> success_list, curvature_descriptor_dicts = values
        >>> curvature_descriptor_dict = curvature_descriptor_dicts[0]
        >>> assert success_list == [True]
        >>> hash_list = [
        >>>     ut.hash_data(curvature_descriptor_dict[scale])
        >>>     for scale in sorted(list(curvature_descriptor_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['zacdsfedcywqdyqozfhdirrcqnypaazw']
    """
    scale_str_list = [
        '%0.04f' % (scale, )
        for scale in scales
    ]
    success_list_ = []
    curvature_descriptor_dicts = []
    for success, curvature in zip(success_list, curvatures):
        success_ = success
        if success:
            curvature_descriptor_list = F.compute_curvature_descriptors(
                curvature,
                curv_length,
                scales,
                num_keypoints,
                uniform,
                feat_dim
            )

            if curvature_descriptor_list is None:
                success_ = False
        else:
            curvature_descriptor_list = None

        curvature_descriptor_dict = None
        if curvature_descriptor_list is not None:
            curvature_descriptor_dict = {
                scale_str: curvature_descriptor
                for scale_str, curvature_descriptor in zip(scale_str_list, curvature_descriptor_list)
            }

        success_list_.append(success_)
        curvature_descriptor_dicts.append(curvature_descriptor_dict)

    return success_list_, curvature_descriptor_dicts


@register_ibs_method
def ibeis_plugin_curvrank_pipeline_compute(ibs, aid_list, config={}):
    r"""
    Args:
        ibs       (IBEISController): IBEIS controller object
        success_list: output of ibeis_plugin_curvrank_outline
        outlines (list of np.ndarray): output of ibeis_plugin_curvrank_outline

    Returns:
        success_list_
        curvature_descriptors

    CommandLine:
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_pipeline_compute
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_pipeline_compute:0
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_pipeline_compute:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> values = ibs.ibeis_plugin_curvrank_pipeline_compute(aid_list)
        >>> success_list, curvature_descriptor_dicts = values
        >>> curvature_descriptor_dict = curvature_descriptor_dicts[0]
        >>> assert success_list == [True]
        >>> hash_list = [
        >>>     ut.hash_data(curvature_descriptor_dict[scale])
        >>>     for scale in sorted(list(curvature_descriptor_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['mkhgqrrkhisuaenxkuxgbbcqpdfpoofp']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> model_type = 'fluke'
        >>> config = {
        >>>     'model_type'     : model_type,
        >>>     'width'          : DEFAULT_WIDTH[model_type],
        >>>     'height'         : DEFAULT_HEIGHT[model_type],
        >>>     'scale'          : DEFAULT_SCALE[model_type],
        >>>     'scales'         : DEFAULT_SCALES[model_type],
        >>>     'allow_diagonal' : DEFAULT_ALLOW_DIAGONAL[model_type],
        >>>     'transpose_dims' : DEFAULT_TRANSPOSE_DIMS[model_type],
        >>> }
        >>> values = ibs.ibeis_plugin_curvrank_pipeline_compute(aid_list, config=config)
        >>> success_list, curvature_descriptor_dicts = values
        >>> curvature_descriptor_dict = curvature_descriptor_dicts[0]
        >>> assert success_list == [True]
        >>> hash_list = [
        >>>     ut.hash_data(curvature_descriptor_dict[scale])
        >>>     for scale in sorted(list(curvature_descriptor_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['zacdsfedcywqdyqozfhdirrcqnypaazw']
    """
    values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list, **config)
    resized_images, resized_masks, pre_transforms = values

    values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks, **config)
    localized_images, localized_masks, loc_transforms = values

    values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms, **config)
    refined_localizations, refined_masks = values

    values = ibs.ibeis_plugin_curvrank_segmentation(refined_localizations, refined_masks, **config)
    segmentations, refined_segmentations = values

    values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks, **config)
    success, starts, ends = values

    args = success, starts, ends, refined_localizations, refined_masks, refined_segmentations
    success, outlines = ibs.ibeis_plugin_curvrank_outline(*args, **config)

    values = ibs.ibeis_plugin_curvrank_trailing_edges(success, outlines, **config)
    success, trailing_edges = values

    values = ibs.ibeis_plugin_curvrank_curvatures(success, trailing_edges, **config)
    success, curvatures = values

    values = ibs.ibeis_plugin_curvrank_curvature_descriptors(success, curvatures, **config)
    success, curvature_descriptors = values

    return success, curvature_descriptors


@register_ibs_method
def ibeis_plugin_curvrank_pipeline_aggregate(ibs, aid_list, success_list,
                                             descriptor_dict_list):
    r"""
    Args:
        ibs       (IBEISController): IBEIS controller object
        success_list: output of ibeis_plugin_curvrank_outline
        outlines (list of np.ndarray): output of ibeis_plugin_curvrank_outline

    Returns:
        success_list_
        curvature_descriptors

    CommandLine:
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_pipeline_aggregate
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_pipeline_aggregate:0
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_pipeline_aggregate:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> values = ibs.ibeis_plugin_curvrank_pipeline_compute(aid_list)
        >>> success_list, curvature_descriptor_dicts = values
        >>> lnbnn_dict = ibs.ibeis_plugin_curvrank_pipeline_aggregate(aid_list, success_list, curvature_descriptor_dicts)
        >>> hash_list = [
        >>>     ut.hash_data(lnbnn_dict[scale])
        >>>     for scale in sorted(list(lnbnn_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['wntkcfldlkhsxvnzvthlenvjrcxblmtd']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> model_type = 'fluke'
        >>> config = {
        >>>     'model_type'     : model_type,
        >>>     'width'          : DEFAULT_WIDTH[model_type],
        >>>     'height'         : DEFAULT_HEIGHT[model_type],
        >>>     'scale'          : DEFAULT_SCALE[model_type],
        >>>     'scales'         : DEFAULT_SCALES[model_type],
        >>>     'allow_diagonal' : DEFAULT_ALLOW_DIAGONAL[model_type],
        >>>     'transpose_dims' : DEFAULT_TRANSPOSE_DIMS[model_type],
        >>> }
        >>> values = ibs.ibeis_plugin_curvrank_pipeline_compute(aid_list, config=config)
        >>> success_list, curvature_descriptor_dicts = values
        >>> lnbnn_dict = ibs.ibeis_plugin_curvrank_pipeline_aggregate(aid_list, success_list, curvature_descriptor_dicts)
        >>> hash_list = [
        >>>     ut.hash_data(lnbnn_dict[scale])
        >>>     for scale in sorted(list(lnbnn_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['ylxevmyeygxlhbcsuwzakfnlisbantdr']
    """
    lnbnn_dict = {}
    zipped = zip(aid_list, success_list, descriptor_dict_list)
    for aid, success, descriptor_dict in zipped:
        if not success:
            continue

        for scale in descriptor_dict:
            if scale not in lnbnn_dict:
                lnbnn_dict[scale] = {
                    'descriptors': [],
                    'aids'       : [],
                }

            descriptors = descriptor_dict[scale]
            aids = [aid] * descriptors.shape[0]

            lnbnn_dict[scale]['descriptors'].append(descriptors)
            lnbnn_dict[scale]['aids'].append(aids)

    for scale in lnbnn_dict:
        descriptors = np.vstack(lnbnn_dict[scale]['descriptors'])
        assert np.allclose(
            np.linalg.norm(descriptors, axis=1),
            np.ones(descriptors.shape[0])
        )

        aids = np.hstack(lnbnn_dict[scale]['aids'])
        lnbnn_dict[scale] = (descriptors, aids, )

    return lnbnn_dict


@register_ibs_method
def ibeis_plugin_curvrank_pipeline(ibs, imageset_rowid=None, aid_list=None,
                                   config={}, use_depc=USE_DEPC, verbose=False):
    r"""
    Args:
        ibs       (IBEISController): IBEIS controller object
        success_list: output of ibeis_plugin_curvrank_outline
        outlines (list of np.ndarray): output of ibeis_plugin_curvrank_outline

    Returns:
        success_list_
        curvature_descriptors

    CommandLine:
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_pipeline
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_pipeline:0
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_pipeline:1
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_pipeline:2
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_pipeline:3

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> lnbnn_dict, aid_list = ibs.ibeis_plugin_curvrank_pipeline(aid_list=aid_list, use_depc=False)
        >>> hash_list = [
        >>>     ut.hash_data(lnbnn_dict[scale])
        >>>     for scale in sorted(list(lnbnn_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['wntkcfldlkhsxvnzvthlenvjrcxblmtd']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> lnbnn_dict, aid_list = ibs.ibeis_plugin_curvrank_pipeline(aid_list=aid_list, use_depc=True)
        >>> hash_list = [
        >>>     ut.hash_data(lnbnn_dict[scale])
        >>>     for scale in sorted(list(lnbnn_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['wntkcfldlkhsxvnzvthlenvjrcxblmtd']

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> model_type = 'fluke'
        >>> config = {
        >>>     'model_type'     : model_type,
        >>>     'width'          : DEFAULT_WIDTH[model_type],
        >>>     'height'         : DEFAULT_HEIGHT[model_type],
        >>>     'scale'          : DEFAULT_SCALE[model_type],
        >>>     'scales'         : DEFAULT_SCALES[model_type],
        >>>     'allow_diagonal' : DEFAULT_ALLOW_DIAGONAL[model_type],
        >>>     'transpose_dims' : DEFAULT_TRANSPOSE_DIMS[model_type],
        >>> }
        >>> lnbnn_dict, aid_list = ibs.ibeis_plugin_curvrank_pipeline(aid_list=aid_list, config=config, use_depc=False)
        >>> hash_list = [
        >>>     ut.hash_data(lnbnn_dict[scale])
        >>>     for scale in sorted(list(lnbnn_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['ylxevmyeygxlhbcsuwzakfnlisbantdr']

    Example3:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> model_type = 'fluke'
        >>> config = DEFAULT_FLUKE_TEST_CONFIG
        >>> lnbnn_dict, aid_list = ibs.ibeis_plugin_curvrank_pipeline(aid_list=aid_list, config=config, use_depc=True)
        >>> hash_list = [
        >>>     ut.hash_data(lnbnn_dict[scale])
        >>>     for scale in sorted(list(lnbnn_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['ylxevmyeygxlhbcsuwzakfnlisbantdr']
    """
    if aid_list is None:
        aid_list = ibs.get_imageset_aids(imageset_rowid)

    # Compute Curvature Descriptors
    if verbose:
        print('\tCompute Curvature Pipeline')

    if use_depc:
        success_list         = ibs.depc_annot.get('curvature_descriptor', aid_list, 'success',    config=config)
        descriptor_dict_list = ibs.depc_annot.get('curvature_descriptor', aid_list, 'descriptor', config=config)
    else:
        values = ibs.ibeis_plugin_curvrank_pipeline_compute(aid_list, config=config)
        success_list, descriptor_dict_list = values

    if verbose:
        print('\tAggregate Pipeline Results')

    lnbnn_dict = ibs.ibeis_plugin_curvrank_pipeline_aggregate(
        aid_list,
        success_list,
        descriptor_dict_list
    )

    return lnbnn_dict, aid_list


@register_ibs_method
def ibeis_plugin_curvrank_scores(ibs, db_aid_list, qr_aid_list, config={},
                                 lnbnn_k=2, verbose=False,
                                 use_names=True, use_depc=USE_DEPC):
    r"""
    CurvRank Example

    Args:
        ibs       (IBEISController): IBEIS controller object
        lnbnn_k   (int): list of image rowids (aids)

    Returns:
        score_dict

    CommandLine:
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_scores
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_scores:0
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_scores:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> db_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Dorsal Database')
        >>> db_aid_list = ibs.get_imageset_aids(db_imageset_rowid)
        >>> qr_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Dorsal Query')
        >>> qr_aid_list = ibs.get_imageset_aids(qr_imageset_rowid)
        >>> score_dict = ibs.ibeis_plugin_curvrank_scores(db_aid_list, qr_aid_list, use_depc=False)
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
        >>> db_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Fluke Database')
        >>> db_aid_list = ibs.get_imageset_aids(db_imageset_rowid)
        >>> qr_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Fluke Query')
        >>> qr_aid_list = ibs.get_imageset_aids(qr_imageset_rowid)
        >>> model_type = 'fluke'
        >>> config = {
        >>>     'model_type'     : model_type,
        >>>     'width'          : DEFAULT_WIDTH[model_type],
        >>>     'height'         : DEFAULT_HEIGHT[model_type],
        >>>     'scale'          : DEFAULT_SCALE[model_type],
        >>>     'scales'         : DEFAULT_SCALES[model_type],
        >>>     'allow_diagonal' : DEFAULT_ALLOW_DIAGONAL[model_type],
        >>>     'transpose_dims' : DEFAULT_TRANSPOSE_DIMS[model_type],
        >>> }
        >>> score_dict = ibs.ibeis_plugin_curvrank_scores(db_aid_list, qr_aid_list, config=config, use_depc=False)
        >>> for key in score_dict:
        >>>     score_dict[key] = round(score_dict[key], 8)
        >>> result = score_dict
        >>> print(result)
        {14: -1.00862974, 7: -0.55433992, 8: -0.70058628, 9: -0.3044969, 10: -0.27739539, 11: -7.8684881, 12: -1.01431028, 13: -1.46861451}
    """
    cache_path = abspath(join(ibs.get_cachedir(), 'curvrank'))
    ut.ensuredir(cache_path)

    if verbose:
        print('Loading database data...')

    values = ibs.ibeis_plugin_curvrank_pipeline(aid_list=db_aid_list, config=config,
                                                verbose=verbose, use_depc=use_depc)
    db_lnbnn_data, _ = values

    if verbose:
        print('Loading query data...')

    values = ibs.ibeis_plugin_curvrank_pipeline(aid_list=qr_aid_list, config=config,
                                                verbose=verbose, use_depc=use_depc)
    qr_lnbnn_data, _ = values

    if verbose:
        print('Loading index for scales...')

    db_annot_uuid_list = ibs.get_annot_uuids(db_aid_list)
    index_hash = ut.hash_data(db_annot_uuid_list)
    index_directory = 'index_%s_aids_%d' % (index_hash, len(db_aid_list), )
    index_path = join(cache_path, index_directory)
    ut.ensuredir(index_path)

    # Build (and cache to disk) LNBNN indexes
    index_filepath_dict = {}
    for scale in db_lnbnn_data:
        index_filepath = join(index_path, '%s.ann' % scale)
        if not exists(index_filepath):
            descriptors, aids = db_lnbnn_data[scale]
            F.build_lnbnn_index(descriptors, index_filepath)
        index_filepath_dict[scale] = index_filepath

    if verbose:
        print('Aggregating scores...')

    # Run LNBNN identification for each scale independently and aggregate
    score_dict = {}
    for scale in index_filepath_dict:
        if scale not in index_filepath_dict:
            continue
        if scale not in db_lnbnn_data:
            continue
        if scale not in qr_lnbnn_data:
            continue

        index_filepath = index_filepath_dict[scale]
        db_descriptors, db_aids = db_lnbnn_data[scale]
        qr_descriptors, qr_aids = qr_lnbnn_data[scale]

        if use_names:
            db_rowids = ibs.get_annot_nids(db_aids)
        else:
            db_rowids = db_aids

        score_dict_ = F.lnbnn_identify(index_filepath, lnbnn_k, qr_descriptors, db_rowids)
        for rowid in score_dict_:
            if rowid not in score_dict:
                score_dict[rowid] = 0.0
            score_dict[rowid] += score_dict_[rowid]

    if verbose:
        print('Returning scores...')

    return score_dict


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis_curvrank._plugin --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
