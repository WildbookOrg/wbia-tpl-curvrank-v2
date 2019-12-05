from __future__ import absolute_import, division, print_function
from ibeis.control import controller_inject  # NOQA
from os.path import abspath, join, exists, split
import ibeis_curvrank.functional as F
from ibeis_curvrank import imutils
# import ibeis.constants as const
from scipy import interpolate
import numpy as np
import utool as ut
import vtool as vt
import datetime
import cv2

# We want to register the depc plugin functions as well, so import it here for IBEIS
import ibeis_curvrank._plugin_depc  # NOQA
from ibeis_curvrank._plugin_depc import DEFAULT_SCALES, _convert_kwargs_config_to_depc_config


_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)
register_api = controller_inject.get_ibeis_flask_api(__name__)


USE_DEPC = True
USE_DEPC_OPTIMIZED = True
ANNOT_INDEX_TREES = 10


FORCE_SERIAL = False
FORCE_SERIAL = FORCE_SERIAL or 'macosx' in ut.get_plat_specifier().lower()
# FORCE_SERIAL = FORCE_SERIAL or const.CONTAINERIZED
CHUNKSIZE = 16


RIGHT_FLIP_LIST = [  # CASE IN-SINSITIVE
    'right',
    'r',
    'dorsal fin right',
    'dorsal_fin_right',
]


HYBRID_FINFINDR_EXTRACTION_FAILURE_CURVRANK_FALLBACK = False


URL_DICT = {
    'dorsal': {
        'localization': 'https://cthulhu.dyn.wildme.io/public/models/curvrank.localization.dorsal.weights.pkl',
        'segmentation': 'https://cthulhu.dyn.wildme.io/public/models/curvrank.segmentation.dorsal.weights.pkl',
    },
    'dorsalfinfindrhybrid': {
        'localization': 'https://cthulhu.dyn.wildme.io/public/models/curvrank.localization.dorsal.weights.pkl',
        'segmentation': 'https://cthulhu.dyn.wildme.io/public/models/curvrank.segmentation.dorsal.weights.pkl',
    },
    'fluke': {
        'localization': None,
        'segmentation': 'https://cthulhu.dyn.wildme.io/public/models/curvrank.segmentation.fluke.weights.pkl',
    },
}

if not HYBRID_FINFINDR_EXTRACTION_FAILURE_CURVRANK_FALLBACK:
    URL_DICT['dorsalfinfindrhybrid']['localization'] = None
    URL_DICT['dorsalfinfindrhybrid']['segmentation'] = None


@register_ibs_method
def ibeis_plugin_curvrank_preprocessing(ibs, aid_list, width=256, height=256, greyscale=False, **kwargs):
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
    ibs._parallel_chips = not FORCE_SERIAL
    config = {
        'greyscale': greyscale,
    }
    image_list = ibs.get_annot_chips(aid_list, config)

    viewpoint_list = ibs.get_annot_viewpoints(aid_list)
    viewpoint_list = [
        None if viewpoint is None else viewpoint.lower()
        for viewpoint in viewpoint_list
    ]
    flip_list = [viewpoint in RIGHT_FLIP_LIST for viewpoint in viewpoint_list]
    height_list = [height] * len(aid_list)
    width_list  = [width]  * len(aid_list)

    zipped = zip(image_list, flip_list, height_list, width_list)

    config_ = {
        'ordered': True,
        'chunksize': CHUNKSIZE,
        'force_serial': ibs.force_serial or FORCE_SERIAL,
        'progkw': {'freq': 10},
    }
    generator = ut.generate2(F.preprocess_image, zipped,
                             nTasks=len(aid_list), **config_)

    resized_images, resized_masks, pre_transforms = [], [], []
    for resized_image, resized_mask, pre_transform in generator:
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
    if model_tag in ['groundtruth']:
        model_url = None
    else:
        model_url = URL_DICT.get(model_type, {}).get(model_tag, None)

    if model_url is None:
        localized_images = resized_images
        localized_masks = resized_masks
        loc_transforms = [np.eye(3, dtype=np.float32)] * len(localized_images)
    else:
        from ibeis_curvrank import localization, model, theano_funcs

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
                                     scale=4, greyscale=False, **kwargs):
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
        >>> aid_list *= 20
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
    config = {
        'greyscale': greyscale,
    }
    image_list = ibs.get_annot_chips(aid_list, config)

    viewpoint_list = ibs.get_annot_viewpoints(aid_list)
    viewpoint_list = [
        None if viewpoint is None else viewpoint.lower()
        for viewpoint in viewpoint_list
    ]
    flip_list = [viewpoint in RIGHT_FLIP_LIST for viewpoint in viewpoint_list]
    scale_list  = [scale]  * len(aid_list)
    height_list = [height] * len(aid_list)
    width_list  = [width]  * len(aid_list)

    zipped = zip(image_list, flip_list, pre_transforms, loc_transforms,
                 scale_list, height_list, width_list)

    config_ = {
        'ordered': True,
        'chunksize': CHUNKSIZE,
        'force_serial': ibs.force_serial or FORCE_SERIAL,
        'futures_threaded': True,
        'progkw': {'freq': 10},
    }
    generator = ut.generate2(F.refine_localization, zipped, nTasks=len(aid_list), **config_)

    refined_localizations, refined_masks = [], []
    for refined_localization, refined_mask in generator:
        refined_localizations.append(refined_localization)
        refined_masks.append(refined_mask)

    return refined_localizations, refined_masks


@register_ibs_method
def ibeis_plugin_curvrank_test_setup_groundtruth(ibs):
    part_rowid_list = ibs.get_valid_part_rowids()
    part_type_list = ibs.get_part_types(part_rowid_list)
    part_contour_list = ibs.get_part_contour(part_rowid_list)
    flag_list = [part_contour.get('contour', None) is not None for part_contour in part_contour_list]

    print('Found %d / %d contours' % (sum(flag_list), len(flag_list), ))

    part_rowid_list = ut.compress(part_rowid_list, flag_list)
    part_type_list = ut.compress(part_type_list, flag_list)
    part_contour_list = ut.compress(part_contour_list, flag_list)

    aid_list = ibs.get_part_aids(part_rowid_list)
    nid_list = ibs.get_annot_nids(aid_list)
    species_list = ibs.get_annot_species_texts(aid_list)
    viewpoint_list = ibs.get_annot_viewpoints(aid_list)
    flag_list = [species == 'fin_dorsal' for species in species_list]

    part_rowid_list = ut.compress(part_rowid_list, flag_list)
    part_type_list = ut.compress(part_type_list, flag_list)
    part_contour_list = ut.compress(part_contour_list, flag_list)
    aid_list = ut.compress(aid_list, flag_list)
    nid_list = ut.compress(nid_list, flag_list)
    species_list = ut.compress(species_list, flag_list)
    viewpoint_list = ut.compress(viewpoint_list, flag_list)

    gid_list = ibs.get_annot_gids(aid_list)
    bbox_list = ibs.get_part_bboxes(part_rowid_list)
    species_list = [
        '%s+%s' % (species, part_type, )
        for species, part_type in zip(species_list, part_type_list)
    ]

    aid_list_ = ibs.add_annots(
        gid_list,
        bbox_list=bbox_list,
        species_list=species_list,
        viewpoint_list=viewpoint_list,
        nid_list=nid_list,
    )
    ibs.delete_parts(ut.flatten(ibs.get_annot_part_rowids(aid_list_)))

    part_rowid_list_ = ibs.add_parts(
        aid_list_,
        bbox_list=bbox_list,
        type_list=part_type_list
    )
    ibs.set_part_contour(part_rowid_list_, part_contour_list)

    all_part_rowid_list_ = ibs.get_annot_part_rowids(aid_list_)
    print('aid_list_ = %r' % (aid_list_, ))
    print('part_rowid_list_ = %r' % (part_rowid_list_, ))
    print('all part_rowid_list_ = %r' % (all_part_rowid_list_, ))

    return aid_list_, part_rowid_list_


@register_ibs_method
def ibeis_plugin_curvrank_test_cleanup_groundtruth(ibs, values=None):
    if values is None:
        values = ibs.ibeis_plugin_curvrank_test_setup_groundtruth()
    aid_list, part_rowid_list = values
    ibs.delete_parts(part_rowid_list)
    ibs.delete_annots(aid_list)


@register_ibs_method
def ibeis_plugin_curvrank_segmentation(ibs, aid_list, refined_localizations, refined_masks,
                                       pre_transforms, loc_transforms,
                                       width=256, height=256, scale=4, model_type='dorsal',
                                       model_tag='segmentation',
                                       groundtruth_radius=25, groundtruth_opacity=0.5,
                                       groundtruth_smooth=True, groundtruth_smooth_margin=0.001,
                                       greyscale=False,
                                       **kwargs):
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
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_segmentation:2
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_segmentation:3

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
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks, pre_transforms, loc_transforms)
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
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks, pre_transforms, loc_transforms, width=width, height=height, scale=scale, model_type=model_type)
        >>> segmentations, refined_segmentations = values
        >>> segmentation = segmentations[0]
        >>> refined_segmentation = refined_segmentations[0]
        >>> assert ut.hash_data(segmentation)         in ['htbsspdnjfchswtcboifeybpkhmbdmms']
        >>> assert ut.hash_data(refined_segmentation) in ['hqngsbdctbjsruuwjhhbuamcbukbyaea']

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list, part_rowid_list = ibs.ibeis_plugin_curvrank_test_setup_groundtruth()
        >>> try:
        >>>     values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list)
        >>>     resized_images, resized_masks, pre_transforms = values
        >>>     values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks, model_tag='groundtruth')
        >>>     localized_images, localized_masks, loc_transforms = values
        >>>     values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms)
        >>>     refined_localizations, refined_masks = values
        >>>     values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks, pre_transforms, loc_transforms, model_tag='groundtruth')
        >>>     segmentations, refined_segmentations = values
        >>>     segmentation = segmentations[0]
        >>>     refined_segmentation = refined_segmentations[0]
        >>>     assert ut.hash_data(segmentation)         in ['owryieckgcmjqptjflybacfcmzgllhiw']
        >>>     assert ut.hash_data(refined_segmentation) in ['ddtxnvyvsskeazpftzlzbobfwxsfrvns']
        >>> finally:
        >>>     ibs.ibeis_plugin_curvrank_test_cleanup_groundtruth()

    Example3:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks, model_type='dorsalfinfindrhybrid')
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms)
        >>> refined_localizations, refined_masks = values
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks, pre_transforms, loc_transforms, model_type='dorsalfinfindrhybrid')
        >>> segmentations, refined_segmentations = values
        >>> segmentation = segmentations[0]
        >>> refined_segmentation = refined_segmentations[0]
        >>> assert segmentation is None
        >>> assert refined_segmentation is None
    """
    if model_tag in ['groundtruth']:
        config = {
            'greyscale': greyscale,
        }
        image_list = ibs.get_annot_chips(aid_list, config)
        viewpoint_list = ibs.get_annot_viewpoints(aid_list)
        viewpoint_list = [
            None if viewpoint is None else viewpoint.lower()
            for viewpoint in viewpoint_list
        ]
        flip_list = [viewpoint in RIGHT_FLIP_LIST for viewpoint in viewpoint_list]

        part_rowids_list = ibs.get_annot_part_rowids(aid_list)
        part_contours_list = list(map(ibs.get_part_contour, part_rowids_list))

        segmentations = []
        refined_segmentations = []

        zipped = zip(aid_list, image_list, flip_list, part_rowids_list, part_contours_list, refined_masks, pre_transforms, loc_transforms)
        for aid, image, flip, part_rowid_list, part_contour_list, refined_mask, pre_transform, loc_transform in zipped:
            part_rowid = None
            part_contour = None

            for part_rowid_, part_contour_ in zip(part_rowid_list, part_contour_list):
                part_contour_ = part_contour_.get('contour', None)
                if part_contour_ is not None:
                    message = 'Cannot have more than one ground-truth contour for aid %r' % (aid, )
                    assert part_rowid is None and part_contour is None, message
                    part_rowid = part_rowid_
                    part_contour = part_contour_

            message = 'Cannot have zero ground-truth contour for aid %r' % (aid, )
            assert part_rowid is not None and part_contour is not None, message

            start = part_contour.get('start', None)
            end = part_contour.get('end', None)
            segment = part_contour.get('segment', [])
            if start is None:
                start = 0
            if end is None:
                end = len(segment)
            segment = segment[start:end]

            canvas_shape = image.shape[:2]
            canvas_h, canvas_w = canvas_shape
            canvas = np.zeros((canvas_h, canvas_w, 1), dtype=np.float64)

            segment_x = np.array(ut.take_column(segment, 'x'))
            segment_y = np.array(ut.take_column(segment, 'y'))

            if groundtruth_smooth:
                try:
                    length = len(segment) * 3
                    mytck, _ = interpolate.splprep([segment_x, segment_y], s=groundtruth_smooth_margin)
                    values = interpolate.splev(np.linspace(0, 1, length), mytck)
                    segment_x, segment_y = values
                except ValueError:
                    pass

            segment_x_ = list(map(int, np.around(segment_x * canvas_w)))
            segment_y_ = list(map(int, np.around(segment_y * canvas_h)))
            zipped = list(zip(segment_x_, segment_y_))

            for radius_ in range(groundtruth_radius, 0, -1):
                radius = radius_ - 1
                opacity = (1.0 - (radius / groundtruth_radius)) ** 3.0
                opacity *= groundtruth_opacity
                color = (opacity, opacity, opacity)
                for x, y in zipped:
                    cv2.circle(canvas, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
            canvas = cv2.blur(canvas, (5, 5))

            refined_segmentation, _ = F.refine_localization(
                canvas, flip, pre_transform, loc_transform,
                scale, height, width
            )
            refined_segmentation[refined_mask < 255] = 0
            refined_segmentation[refined_segmentation < 0] = 0

            segmentation_ = imutils.refine_segmentation(refined_segmentation, 1.0 / scale)
            segmentation_ = segmentation_.reshape((height, width, 1))
            segmentation_ = segmentation_.astype(np.float64)

            segmentations.append(segmentation_)
            refined_segmentations.append(refined_segmentation)
    else:
        model_url = URL_DICT.get(model_type, {}).get(model_tag, None)
        if model_url is None:
            segmentation_ = np.zeros((height, width, 1), dtype=np.float64)
            refined_segmentation = np.zeros((height * scale, width * scale, 1), dtype=np.float64)

            segmentations = [segmentation_] * len(aid_list)
            refined_segmentations = [refined_segmentation] * len(aid_list)
        else:
            from ibeis_curvrank import segmentation, model, theano_funcs
            weight_filepath = ut.grab_file_url(model_url, appname='ibeis_curvrank', check_hash=True)

            segmentation_layers = segmentation.build_model_batchnorm_full((None, 3, height, width))

            # I am not sure these are the correct args to load_weights
            model.load_weights(segmentation_layers['seg_out'], weight_filepath)
            segmentation_func = theano_funcs.create_segmentation_func(segmentation_layers)
            values = F.segment_contour(refined_localizations, refined_masks, scale,
                                       height, width, segmentation_func)

            segmentations, refined_segmentations = values

    return segmentations, refined_segmentations


def ibeis_plugin_curvrank_keypoints_worker(model_type, segmentation,
                                           localized_mask):
    if model_type in ['dorsal', 'dorsalfinfindrhybrid']:
        from ibeis_curvrank.dorsal_utils import find_dorsal_keypoints as find_func
    else:
        from ibeis_curvrank.dorsal_utils import find_fluke_keypoints as find_func

    try:
        start, end = F.find_keypoints(
            find_func,
            segmentation,
            localized_mask
        )
    except ZeroDivisionError:
        start = None
        end = None

    success = start is not None and end is not None

    return success, start, end


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
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_keypoints:2
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_keypoints:3

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
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks, pre_transforms, loc_transforms)
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
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks, pre_transforms, loc_transforms, width=width, height=height, scale=scale, model_type=model_type)
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks, model_type=model_type)
        >>> success_list, starts, ends = values
        >>> start = tuple(starts[0])
        >>> end = tuple(ends[0])
        >>> assert success_list == [True]
        >>> assert start == (56, 8)
        >>> assert end   == (59, 358)

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list, part_rowid_list = ibs.ibeis_plugin_curvrank_test_setup_groundtruth()
        >>> try:
        >>>     values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list)
        >>>     resized_images, resized_masks, pre_transforms = values
        >>>     values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks, model_tag='groundtruth')
        >>>     localized_images, localized_masks, loc_transforms = values
        >>>     values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms)
        >>>     refined_localizations, refined_masks = values
        >>>     values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks, pre_transforms, loc_transforms, model_tag='groundtruth')
        >>>     segmentations, refined_segmentations = values
        >>>     values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks)
        >>>     success_list, starts, ends = values
        >>>     start = tuple(starts[0])
        >>>     end = tuple(ends[0])
        >>>     assert success_list == [True]
        >>>     assert start == (238, 3)
        >>>     assert end   == (182, 253)
        >>> finally:
        >>>     ibs.ibeis_plugin_curvrank_test_cleanup_groundtruth()

    Example3:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks, model_type='dorsalfinfindrhybrid')
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms)
        >>> refined_localizations, refined_masks = values
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks, pre_transforms, loc_transforms, model_type='dorsalfinfindrhybrid')
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks, model_type='dorsalfinfindrhybrid')
        >>> success_list, starts, ends = values
        >>> start = tuple(starts[0])
        >>> end = tuple(ends[0])
        >>> assert success_list == [True]
        >>> assert start == (None, None)
        >>> assert end   == (None, None)
    """
    num_total = len(segmentations)

    if model_type in ['dorsalfinfindrhybrid'] and not HYBRID_FINFINDR_EXTRACTION_FAILURE_CURVRANK_FALLBACK:
        success_list = [False] * num_total
        starts       = [(None, None)] * num_total
        ends         = [(None, None)] * num_total
    else:
        model_type_list = [model_type] * num_total

        zipped = zip(model_type_list, segmentations, localized_masks)

        config_ = {
            'ordered': True,
            'chunksize': CHUNKSIZE,
            'force_serial': ibs.force_serial or FORCE_SERIAL,
            'progkw': {'freq': 10},
        }
        generator = ut.generate2(ibeis_plugin_curvrank_keypoints_worker, zipped,
                                 nTasks=num_total, **config_)

        starts, ends, success_list = [], [], []
        for success, start, end in generator:
            success_list.append(success)
            starts.append(start)
            ends.append(end)

    return success_list, starts, ends


def ibeis_plugin_curvrank_outline_worker(model_type, success, start, end, refined_loc,
                                         refined_mask, refined_seg, scale, allow_diagonal):
    if model_type in ['dorsal', 'dorsalfinfindrhybrid']:
        from ibeis_curvrank.dorsal_utils import dorsal_cost_func as cost_func
    else:
        from ibeis_curvrank.dorsal_utils import fluke_cost_func as cost_func

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

    return success_, outline


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
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_outline:2
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_outline:3

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
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks, pre_transforms, loc_transforms)
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
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks, pre_transforms, loc_transforms, width=width, height=height, scale=scale, model_type=model_type)
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks, model_type=model_type)
        >>> success_list, starts, ends = values
        >>> args = success_list, starts, ends, refined_localizations, refined_masks, refined_segmentations
        >>> success_list, outlines = ibs.ibeis_plugin_curvrank_outline(*args, scale=scale, model_type=model_type, allow_diagonal=allow_diagonal)
        >>> outline = outlines[0]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(outline) in ['qqvetxfhhipfuqneuinwrvcztkjlfoak']

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list, part_rowid_list = ibs.ibeis_plugin_curvrank_test_setup_groundtruth()
        >>> try:
        >>>     values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list)
        >>>     resized_images, resized_masks, pre_transforms = values
        >>>     values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks, model_tag='groundtruth')
        >>>     localized_images, localized_masks, loc_transforms = values
        >>>     values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms)
        >>>     refined_localizations, refined_masks = values
        >>>     values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks, pre_transforms, loc_transforms, model_tag='groundtruth')
        >>>     segmentations, refined_segmentations = values
        >>>     values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks)
        >>>     success_list, starts, ends = values
        >>>     args = success_list, starts, ends, refined_localizations, refined_masks, refined_segmentations
        >>>     success_list, outlines = ibs.ibeis_plugin_curvrank_outline(*args)
        >>>     outline = outlines[0]
        >>>     assert success_list == [True]
        >>>     assert ut.hash_data(outline) in ['ykbndjqawiersnktufkmdtbwsfuexyeg']
        >>> finally:
        >>>     ibs.ibeis_plugin_curvrank_test_cleanup_groundtruth()

    Example3:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks, model_type='dorsalfinfindrhybrid')
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms)
        >>> refined_localizations, refined_masks = values
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks, pre_transforms, loc_transforms, model_type='dorsalfinfindrhybrid')
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks, model_type='dorsalfinfindrhybrid')
        >>> success_list, starts, ends = values
        >>> args = success_list, starts, ends, refined_localizations, refined_masks, refined_segmentations
        >>> success_list, outlines = ibs.ibeis_plugin_curvrank_outline(*args, model_type='dorsalfinfindrhybrid')
        >>> outline = outlines[0]
        >>> assert success_list == [True]
        >>> assert outline is None
    """
    num_total = len(success_list)

    if model_type in ['dorsalfinfindrhybrid'] and not HYBRID_FINFINDR_EXTRACTION_FAILURE_CURVRANK_FALLBACK:
        success_list_ = [False] * num_total
        outlines      = [None] * num_total
    else:
        model_type_list     = [model_type]     * num_total
        scale_list          = [scale]          * num_total
        allow_diagonal_list = [allow_diagonal] * num_total

        zipped = zip(model_type_list, success_list, starts, ends, refined_localizations,
                     refined_masks, refined_segmentations, scale_list, allow_diagonal_list)

        config_ = {
            'ordered': True,
            'chunksize': CHUNKSIZE,
            # 'force_serial': ibs.force_serial or FORCE_SERIAL,
            # 'futures_threaded': True,
            'force_serial': True,
            'progkw': {'freq': 10},
        }
        generator = ut.generate2(ibeis_plugin_curvrank_outline_worker, zipped,
                                 nTasks=num_total, **config_)

        success_list_, outlines = [], []
        for success_, outline in generator:
            success_list_.append(success_)
            outlines.append(outline)

    return success_list_, outlines


def ibeis_plugin_curvrank_trailing_edges_worker(success, outline):
    from ibeis_curvrank.dorsal_utils import separate_leading_trailing_edges

    success_ = success
    if success:
        values = F.separate_edges(separate_leading_trailing_edges, outline)
        _, trailing_edge = values

        if trailing_edge is None:
            success_ = False
    else:
        trailing_edge = None

    return success_, trailing_edge


@register_ibs_method
def ibeis_plugin_curvrank_trailing_edges(ibs, aid_list, success_list, outlines,
                                         model_type='dorsal', width=256, height=256,
                                         scale=4, finfindr_smooth=True,
                                         finfindr_smooth_margin=0.001, **kwargs):
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
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_trailing_edges:2

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
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks, pre_transforms, loc_transforms)
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks)
        >>> success_list, starts, ends = values
        >>> args = success_list, starts, ends, refined_localizations, refined_masks, refined_segmentations
        >>> success_list, outlines = ibs.ibeis_plugin_curvrank_outline(*args)
        >>> values = ibs.ibeis_plugin_curvrank_trailing_edges(aid_list, success_list, outlines)
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
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks, pre_transforms, loc_transforms, width=width, height=height, scale=scale, model_type=model_type)
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks, model_type=model_type)
        >>> success_list, starts, ends = values
        >>> args = success_list, starts, ends, refined_localizations, refined_masks, refined_segmentations
        >>> values = ibs.ibeis_plugin_curvrank_outline(*args, scale=scale, model_type=model_type, allow_diagonal=allow_diagonal)
        >>> success_list, outlines = values
        >>> values = ibs.ibeis_plugin_curvrank_trailing_edges(aid_list, success_list, outlines, model_type=model_type)
        >>> success_list, trailing_edges = values
        >>> assert success_list == [True]
        >>> assert ut.hash_data(outlines) == ut.hash_data(trailing_edges)

    Example2:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks, model_type='dorsalfinfindrhybrid')
        >>> localized_images, localized_masks, loc_transforms = values
        >>> values = ibs.ibeis_plugin_curvrank_refinement(aid_list, pre_transforms, loc_transforms)
        >>> refined_localizations, refined_masks = values
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks, pre_transforms, loc_transforms, model_type='dorsalfinfindrhybrid')
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks, model_type='dorsalfinfindrhybrid')
        >>> success_list, starts, ends = values
        >>> args = success_list, starts, ends, refined_localizations, refined_masks, refined_segmentations
        >>> success_list, outlines = ibs.ibeis_plugin_curvrank_outline(*args, model_type='dorsalfinfindrhybrid')
        >>> values = ibs.ibeis_plugin_curvrank_trailing_edges(aid_list, success_list, outlines, model_type='dorsalfinfindrhybrid')
        >>> success_list, trailing_edges = values
        >>> trailing_edge = trailing_edges[0]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(trailing_edge[0]) in ['fczhuzbqtxjhctkymrlugyyypgrcegfy']
    """
    model_tpe_list = ['dorsal']
    if HYBRID_FINFINDR_EXTRACTION_FAILURE_CURVRANK_FALLBACK:
        model_tpe_list.append('dorsalfinfindrhybrid')

    if model_type in model_tpe_list:
        zipped = zip(success_list, outlines)

        config_ = {
            'ordered': True,
            'chunksize': CHUNKSIZE,
            'force_serial': ibs.force_serial or FORCE_SERIAL,
            'progkw': {'freq': 10},
        }
        generator = ut.generate2(ibeis_plugin_curvrank_trailing_edges_worker, zipped,
                                 nTasks=len(success_list), **config_)

        success_list_ = []
        trailing_edges = []
        for success_, trailing_edge in generator:
            success_list_.append(success_)
            trailing_edges.append(trailing_edge)

    if model_type in ['dorsalfinfindrhybrid']:
        from numpy.linalg import inv
        from ibeis_curvrank import affine

        # backup_success_list_ = success_list_[:]
        # backup_trailing_edges = trailing_edges[:]
        backup_success_list = [False] * len(aid_list)
        backup_trailing_edges = [None] * len(aid_list)

        # Get original chips and viewpoints
        chip_list = ibs.get_annot_chips(aid_list)
        shape_list = [chip.shape[:2] for chip in chip_list]
        viewpoint_list = ibs.get_annot_viewpoints(aid_list)
        viewpoint_list = [
            None if viewpoint is None else viewpoint.lower()
            for viewpoint in viewpoint_list
        ]
        flip_list = [viewpoint in RIGHT_FLIP_LIST for viewpoint in viewpoint_list]

        # Get CurvRank primitives
        values = ibs.ibeis_plugin_curvrank_preprocessing(aid_list)
        resized_images, resized_masks, pre_transforms = values
        values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks)
        localized_images, localized_masks, loc_transforms = values

        # Get FinfindR primitives
        annot_hash_data_list = ibs.depc_annot.get('FinfindrFeature', aid_list, 'response')
        finfindr_chip_path_list = ibs.finfindr_annot_chip_fpath_from_aid(aid_list)
        finfindr_chips = [
            cv2.imread(finfindr_chip_path)
            for finfindr_chip_path in finfindr_chip_path_list
        ]
        finfindr_shape_list = [finfindr_chip.shape[:2] for finfindr_chip in finfindr_chips]

        success_list_ = []
        trailing_edges = []
        zipped = zip(
            aid_list,
            annot_hash_data_list,
            shape_list,
            finfindr_shape_list,
            flip_list,
            pre_transforms,
            loc_transforms,
            backup_success_list,
            backup_trailing_edges,
        )
        for values in zipped:
            aid, annot_hash_data, shape, finfindr_shape, flip, pre_transform, loc_transform, backup_success, backup_trailing_edge  = values

            if annot_hash_data is None:
                annot_hash_data = {}

            coordinates = annot_hash_data.get('coordinates', None)

            if coordinates is None:
                print('[Hybrid] Using CurvRank trailing edge as a backup for AID %r because FinfindR failed to extract' % (aid, ))

                backup_aid_list = [aid]
                values = ibs.ibeis_plugin_curvrank_preprocessing(backup_aid_list)
                backup_resized_images_, backup_resized_masks, backup_pre_transforms = values
                values = ibs.ibeis_plugin_curvrank_localization(backup_resized_images_, backup_resized_masks)
                backup_localized_images, backup_localized_masks, backup_loc_transforms = values
                values = ibs.ibeis_plugin_curvrank_refinement(backup_aid_list, backup_pre_transforms, backup_localized_images)
                backup_refined_localizations, backup_refined_masks = values
                values = ibs.ibeis_plugin_curvrank_segmentation(backup_aid_list, backup_refined_localizations, backup_refined_masks, backup_pre_transforms, backup_loc_transforms)
                backup_segmentations, backup_refined_segmentations = values
                values = ibs.ibeis_plugin_curvrank_keypoints(backup_segmentations, backup_localized_masks)
                backup_success_list, backup_starts, backup_ends = values
                args = backup_success_list, backup_starts, backup_ends, backup_refined_localizations, backup_refined_masks, backup_refined_segmentations
                backup_success_list, backup_outlines = ibs.ibeis_plugin_curvrank_outline(*args)
                values = ibs.ibeis_plugin_curvrank_trailing_edges(backup_aid_list, backup_success_list, backup_outlines)
                backup_success_list, backup_trailing_edges = values
                success = backup_success_list[0]
                trailing_edge = backup_trailing_edges[0]
            else:
                success = True

                T10 = affine.build_downsample_matrix(height, width)
                T21 = loc_transform
                T32 = affine.build_upsample_matrix(height, width)
                T43 = cv2.invertAffineTransform(pre_transform[:2])
                T70 = affine.build_scale_matrix(scale)

                T70i = cv2.invertAffineTransform(T70[:2])
                A = affine.multiply_matrices([T43, T32, T21, T10, T70i])
                A_ = inv(A)

                h, w = shape
                finfindr_h, finfindr_w = finfindr_shape

                trailing_edge = []
                last_point = None
                for x, y in coordinates:
                    if flip:
                        x = finfindr_w - x
                    x = (x / finfindr_w) * w
                    y = (y / finfindr_h) * h
                    point = np.array([[x, y]])
                    point = vt.transform_points_with_homography(A_, point.T).T
                    x, y = point[0]
                    x = int(np.around(x))
                    y = int(np.around(y))
                    point = [y, x]
                    if point != last_point:
                        trailing_edge.append(point)
                    last_point = point
                trailing_edge = np.array(trailing_edge)

                if finfindr_smooth:
                    try:
                        length = len(trailing_edge) * 3
                        # length = 10000
                        x = trailing_edge[:, 0]
                        y = trailing_edge[:, 1]
                        mytck, _ = interpolate.splprep([x, y], s=finfindr_smooth_margin)
                        x_, y_ = interpolate.splev(np.linspace(0, 1, length), mytck)

                        trailing_edge_ = np.array(list(zip(x_, y_)))
                        trailing_edge_ = np.around(trailing_edge_).astype(trailing_edge.dtype)

                        new_trailing_edge = []
                        last_point = None
                        for point in trailing_edge_:
                            point = list(point)
                            if point != last_point:
                                new_trailing_edge.append(point)
                            last_point = point
                        trailing_edge = np.array(new_trailing_edge)
                    except:
                        pass

                # Make sure the first point of the trailing_edge is the top of the fin
                first_y = trailing_edge[0][0]
                last_y  = trailing_edge[-1][0]
                if last_y < first_y:
                    trailing_edge = trailing_edge[::-1]

            success_list_.append(success)
            trailing_edges.append(trailing_edge)

    if model_type in ['fluke']:
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
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks, pre_transforms, loc_transforms)
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks)
        >>> success_list, starts, ends = values
        >>> args = success_list, starts, ends, refined_localizations, refined_masks, refined_segmentations
        >>> success_list, outlines = ibs.ibeis_plugin_curvrank_outline(*args)
        >>> values = ibs.ibeis_plugin_curvrank_trailing_edges(aid_list, success_list, outlines)
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
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks, pre_transforms, loc_transforms, width=width, height=height, scale=scale, model_type=model_type)
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks, model_type=model_type)
        >>> success_list, starts, ends = values
        >>> args = success_list, starts, ends, refined_localizations, refined_masks, refined_segmentations
        >>> values = ibs.ibeis_plugin_curvrank_outline(*args, scale=scale, model_type=model_type, allow_diagonal=allow_diagonal)
        >>> success_list, outlines = values
        >>> values = ibs.ibeis_plugin_curvrank_trailing_edges(aid_list, success_list, outlines, model_type=model_type)
        >>> success_list, trailing_edges = values
        >>> values = ibs.ibeis_plugin_curvrank_curvatures(success_list, trailing_edges, scales=scales, transpose_dims=transpose_dims)
        >>> success_list, curvatures = values
        >>> curvature = curvatures[0]
        >>> assert success_list == [True]
        >>> assert ut.hash_data(curvature) in ['dpbusvatdgcdblmtwvodvlsnjuffdylp']
    """
    scales_list = [scales] * len(success_list)
    transpose_dims_list = [transpose_dims] * len(success_list)
    zipped = zip(success_list, trailing_edges, scales_list, transpose_dims_list)

    config_ = {
        'ordered': True,
        'chunksize': CHUNKSIZE,
        'force_serial': ibs.force_serial or FORCE_SERIAL,
        'progkw': {'freq': 10},
    }
    generator = ut.generate2(ibeis_plugin_curvrank_curvatures_worker, zipped,
                             nTasks=len(success_list), **config_)

    success_list_ = []
    curvatures = []
    for success_, curvature in generator:
        success_list_.append(success_)
        curvatures.append(curvature)

    return success_list_, curvatures


def ibeis_plugin_curvrank_curvatures_worker(success, trailing_edge, scales,
                                            transpose_dims):
    success_ = success
    if success:
        curvature = F.compute_curvature(trailing_edge, scales, transpose_dims)

        if curvature is None:
            success_ = False
    else:
        curvature = None

    return success_, curvature


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
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks, pre_transforms, loc_transforms)
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks)
        >>> success_list, starts, ends = values
        >>> args = success_list, starts, ends, refined_localizations, refined_masks, refined_segmentations
        >>> success_list, outlines = ibs.ibeis_plugin_curvrank_outline(*args)
        >>> values = ibs.ibeis_plugin_curvrank_trailing_edges(aid_list, success_list, outlines)
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
        >>> values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks, pre_transforms, loc_transforms, width=width, height=height, scale=scale, model_type=model_type)
        >>> segmentations, refined_segmentations = values
        >>> values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks, model_type=model_type)
        >>> success_list, starts, ends = values
        >>> args = success_list, starts, ends, refined_localizations, refined_masks, refined_segmentations
        >>> values = ibs.ibeis_plugin_curvrank_outline(*args, scale=scale, model_type=model_type, allow_diagonal=allow_diagonal)
        >>> success_list, outlines = values
        >>> values = ibs.ibeis_plugin_curvrank_trailing_edges(aid_list, success_list, outlines, model_type=model_type)
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
    curv_length_list   = [curv_length]   * len(success_list)
    scales_list        = [scales]        * len(success_list)
    num_keypoints_list = [num_keypoints] * len(success_list)
    uniform_list       = [uniform]       * len(success_list)
    feat_dim_list      = [feat_dim]      * len(success_list)

    zipped = zip(success_list, curvatures, curv_length_list, scales_list, num_keypoints_list,
                 uniform_list, feat_dim_list)

    config_ = {
        'ordered': True,
        'chunksize': CHUNKSIZE,
        'force_serial': ibs.force_serial or FORCE_SERIAL,
        'progkw': {'freq': 10},
    }
    generator = ut.generate2(ibeis_plugin_curvrank_curvature_descriptors_worker, zipped,
                             nTasks=len(success_list), **config_)

    success_list_ = []
    curvature_descriptor_dicts = []
    for success_, curvature_descriptor_dict in generator:
        success_list_.append(success_)
        curvature_descriptor_dicts.append(curvature_descriptor_dict)

    return success_list_, curvature_descriptor_dicts


def ibeis_plugin_curvrank_curvature_descriptors_worker(success, curvature, curv_length,
                                                       scales, num_keypoints, uniform,
                                                       feat_dim):
    scale_str_list = [
        '%0.04f' % (scale, )
        for scale in scales
    ]

    success_ = success
    if success:
        try:
            curvature_descriptor_list = F.compute_curvature_descriptors(
                curvature,
                curv_length,
                scales,
                num_keypoints,
                uniform,
                feat_dim
            )
        except:
            curvature_descriptor_list = None

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

    return success_, curvature_descriptor_dict


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
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_pipeline_compute:2
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_pipeline_compute:3

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

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> aid_list *= 20
        >>> values = ibs.ibeis_plugin_curvrank_pipeline_compute(aid_list)
        >>> success_list, curvature_descriptor_dicts = values
        >>> success_list = success_list[:1]
        >>> curvature_descriptor_dicts = curvature_descriptor_dicts[:1]
        >>> curvature_descriptor_dict = curvature_descriptor_dicts[0]
        >>> assert success_list == [True]
        >>> hash_list = [
        >>>     ut.hash_data(curvature_descriptor_dict[scale])
        >>>     for scale in sorted(list(curvature_descriptor_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['mkhgqrrkhisuaenxkuxgbbcqpdfpoofp']

    Example3:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> from ibeis_curvrank._plugin_depc import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> aid_list *= 20
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
        >>> success_list = success_list[:1]
        >>> curvature_descriptor_dicts = curvature_descriptor_dicts[:1]
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

    values = ibs.ibeis_plugin_curvrank_segmentation(aid_list, refined_localizations, refined_masks, pre_transforms, loc_transforms, **config)
    segmentations, refined_segmentations = values

    values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks, **config)
    success, starts, ends = values

    args = success, starts, ends, refined_localizations, refined_masks, refined_segmentations
    success, outlines = ibs.ibeis_plugin_curvrank_outline(*args, **config)

    values = ibs.ibeis_plugin_curvrank_trailing_edges(aid_list, success, outlines, **config)
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
                                   config={}, use_depc=USE_DEPC,
                                   use_depc_optimized=USE_DEPC_OPTIMIZED,
                                   verbose=False):
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
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_pipeline:4
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_pipeline:5

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
        >>> lnbnn_dict, aid_list = ibs.ibeis_plugin_curvrank_pipeline(aid_list=aid_list, use_depc=True, use_depc_optimized=False)
        >>> hash_list = [
        >>>     ut.hash_data(lnbnn_dict[scale])
        >>>     for scale in sorted(list(lnbnn_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['wntkcfldlkhsxvnzvthlenvjrcxblmtd']

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_curvrank._plugin import *  # NOQA
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> lnbnn_dict, aid_list = ibs.ibeis_plugin_curvrank_pipeline(aid_list=aid_list, use_depc=True, use_depc_optimized=True)
        >>> hash_list = [
        >>>     ut.hash_data(lnbnn_dict[scale])
        >>>     for scale in sorted(list(lnbnn_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['wntkcfldlkhsxvnzvthlenvjrcxblmtd']

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

    Example4:
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
        >>> lnbnn_dict, aid_list = ibs.ibeis_plugin_curvrank_pipeline(aid_list=aid_list, config=config, use_depc=True, use_depc_optimized=False)
        >>> hash_list = [
        >>>     ut.hash_data(lnbnn_dict[scale])
        >>>     for scale in sorted(list(lnbnn_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['ylxevmyeygxlhbcsuwzakfnlisbantdr']

    Example5:
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
        >>> lnbnn_dict, aid_list = ibs.ibeis_plugin_curvrank_pipeline(aid_list=aid_list, config=config, use_depc=True, use_depc_optimized=True)
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
        config_ = _convert_kwargs_config_to_depc_config(config)
        table_name = 'curvature_descriptor_optimized' if use_depc_optimized else 'curvature_descriptor'
        success_list         = ibs.depc_annot.get(table_name, aid_list, 'success',    config=config_)
        descriptor_dict_list = ibs.depc_annot.get(table_name, aid_list, 'descriptor', config=config_)
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
def ibeis_plugin_curvrank_scores(ibs, db_aid_list, qr_aids_list, config={},
                                 lnbnn_k=2, verbose=False,
                                 use_names=True,
                                 minimum_score=-1e-5,
                                 use_depc=USE_DEPC,
                                 use_depc_optimized=USE_DEPC_OPTIMIZED):
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
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_scores:2
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_scores:3
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_scores:4
        python -m ibeis_curvrank._plugin --test-ibeis_plugin_curvrank_scores:5

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
        >>> score_dict_iter = ibs.ibeis_plugin_curvrank_scores(db_aid_list, [qr_aid_list], use_depc=False)
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
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> db_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Dorsal Database')
        >>> db_aid_list = ibs.get_imageset_aids(db_imageset_rowid)
        >>> qr_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Dorsal Query')
        >>> qr_aid_list = ibs.get_imageset_aids(qr_imageset_rowid)
        >>> score_dict_iter = ibs.ibeis_plugin_curvrank_scores(db_aid_list, [qr_aid_list], use_depc=True, use_depc_optimized=False)
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
        >>> import ibeis
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> db_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Dorsal Database')
        >>> db_aid_list = ibs.get_imageset_aids(db_imageset_rowid)
        >>> qr_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Dorsal Query')
        >>> qr_aid_list = ibs.get_imageset_aids(qr_imageset_rowid)
        >>> score_dict_iter = ibs.ibeis_plugin_curvrank_scores(db_aid_list, [qr_aid_list], use_depc=True, use_depc_optimized=True)
        >>> score_dict_list = list(score_dict_iter)
        >>> qr_aid_list, score_dict = score_dict_list[0]
        >>> for key in score_dict:
        >>>     score_dict[key] = round(score_dict[key], 8)
        >>> result = score_dict
        >>> print(result)
        {1: -31.81339289, 2: -3.7092349, 3: -4.95274189}

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
        >>> score_dict_iter = ibs.ibeis_plugin_curvrank_scores(db_aid_list, [qr_aid_list], config=config, use_depc=False)
        >>> score_dict_list = list(score_dict_iter)
        >>> qr_aid_list, score_dict = score_dict_list[0]
        >>> for key in score_dict:
        >>>     score_dict[key] = round(score_dict[key], 8)
        >>> result = score_dict
        >>> print(result)
        {14: -1.00862974, 7: -0.55433992, 8: -0.70058628, 9: -0.3044969, 10: -0.27739539, 11: -7.8684881, 12: -1.01431028, 13: -1.46861451}

    Example4:
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
        >>> score_dict_iter = ibs.ibeis_plugin_curvrank_scores(db_aid_list, [qr_aid_list], config=config, use_depc=True, use_depc_optimized=False)
        >>> score_dict_list = list(score_dict_iter)
        >>> qr_aid_list, score_dict = score_dict_list[0]
        >>> for key in score_dict:
        >>>     score_dict[key] = round(score_dict[key], 8)
        >>> result = score_dict
        >>> print(result)
        {14: -1.00862974, 7: -0.55433992, 8: -0.70058628, 9: -0.3044969, 10: -0.27739539, 11: -7.8684881, 12: -1.01431028, 13: -1.46861451}

    Example5:
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
        >>> score_dict_iter = ibs.ibeis_plugin_curvrank_scores(db_aid_list, [qr_aid_list], config=config, use_depc=True, use_depc_optimized=True)
        >>> score_dict_list = list(score_dict_iter)
        >>> qr_aid_list, score_dict = score_dict_list[0]
        >>> for key in score_dict:
        >>>     score_dict[key] = round(score_dict[key], 8)
        >>> result = score_dict
        >>> print(result)
        {14: -1.00862974, 7: -0.55433992, 8: -0.70058628, 9: -0.3044969, 10: -0.27739539, 11: -7.8684881, 12: -1.01431028, 13: -1.46861451}
    """
    cache_path = abspath(join(ibs.get_cachedir(), 'curvrank'))
    ut.ensuredir(cache_path)

    FUTURE_PREFIX = '__future__'
    TTL_HOUR_DELETE = 3 * 24
    TTL_HOUR_PREVIOUS = 1 * 24

    use_daily_cache = config.pop('use_daily_cache', False)
    daily_cache_tag = config.pop('daily_cache_tag', 'global')
    force_cache_recompute = config.pop('force_cache_recompute', False)

    args = (use_daily_cache, daily_cache_tag, force_cache_recompute, )
    print('CurvRank cache config:\n\tuse_daily_cache = %r\n\tdaily_cache_tag = %r\n\tforce_cache_recompute = %r\n\t' % args)
    print('CurvRank algo  config: %s' % (ut.repr3(config), ))

    config_hash = ut.hash_data(ut.repr3(config))
    now = datetime.datetime.now()
    timestamp_fmtstr = '%Y-%m-%d-%H-%M-%S'
    timestamp = now.strftime(timestamp_fmtstr)

    daily_cache_tag = str(daily_cache_tag)
    if daily_cache_tag in ['global']:
        qr_aid_list = ut.flatten(qr_aids_list)
        qr_species_set = set(ibs.get_annot_species_texts(qr_aid_list))
        qr_species_str = '-'.join(sorted(qr_species_set))
        daily_index_hash = 'daily-global-%s' % (qr_species_str)
    else:
        daily_index_hash = 'daily-tag-%s' % (daily_cache_tag)

    with ut.Timer('Clearing old caches (TTL = %d hours)' % (TTL_HOUR_DELETE, )):

        delete = datetime.timedelta(hours=TTL_HOUR_DELETE)
        past_delete = now - delete

        previous = datetime.timedelta(hours=TTL_HOUR_PREVIOUS)
        past_previous = now - previous

        available_previous_list = []
        for path in ut.glob(join(cache_path, 'index_*')):
            try:
                directory = split(path)[1]
                date_str = directory.split('_')[1]
                then = datetime.datetime.strptime(date_str, timestamp_fmtstr)
                print('Checking %r (%r)' % (directory, then, ))

                if then < past_delete:
                    print('\ttoo old, deleting %r...' % (path, ))
                    ut.delete(path)
                else:
                    if past_previous <= then:
                        if daily_index_hash in directory:
                            available_previous_list.append(directory)
                    delta = then - past_delete
                    hours = delta.total_seconds() / 60 / 60
                    print('\tkeeping cache for %0.2f more hours...' % (hours, ))
            except:
                print('\tinvalid (parse error), deleting %r...' % (path, ))
                ut.delete(path)

        # Check for any FUTURE_PREFIX folders that are too old (due to an error) and need to be deleted
        for path in ut.glob(join(cache_path, '%sindex_*' % (FUTURE_PREFIX, ))):
            try:
                directory = split(path)[1]
                directory = directory.replace(FUTURE_PREFIX, '')
                date_str = directory.split('_')[1]
                then = datetime.datetime.strptime(date_str, timestamp_fmtstr)
                print('Checking %r (%r)' % (directory, then, ))

                if then < past_delete:
                    print('\ttoo old, deleting %r...' % (path, ))
                    ut.delete(path)
            except:
                print('\tinvalid (parse error), deleting %r...' % (path, ))
                ut.delete(path)

        available_previous_list = sorted(available_previous_list)
        args = (ut.repr3(available_previous_list), )
        print('\nAvailable previous cached: %s' % args)

    if use_daily_cache:
        if force_cache_recompute or len(available_previous_list) == 0:
            args = (timestamp, daily_index_hash, config_hash, )
            index_directory = 'index_%s_hash_%s_config_%s' % args
            print('Using daily index (recompute = %r): %r' % (force_cache_recompute, index_directory, ))
        else:
            index_directory = available_previous_list[-1]
            print('Using the most recent available index: %r' % (index_directory, ))
    else:
        all_aid_list = ut.flatten(qr_aids_list) + db_aid_list
        all_annot_uuid_list = ibs.get_annot_uuids(sorted(all_aid_list))
        index_hash = ut.hash_data(all_annot_uuid_list)

        args = (timestamp, index_hash, config_hash, )
        index_directory = 'index_%s_hash_%s_config_%s' % args
        print('Using hashed index: %r' % (index_directory, ))

    index_path = join(cache_path, index_directory)

    with ut.Timer('Loading query'):
        scale_set = set([])
        qr_lnbnn_data_list = []
        for qr_aid_list in ut.ProgressIter(qr_aids_list, lbl='CurvRank Query LNBNN', freq=1000):
            values = ibs.ibeis_plugin_curvrank_pipeline(aid_list=qr_aid_list, config=config,
                                                        verbose=verbose, use_depc=use_depc,
                                                        use_depc_optimized=use_depc_optimized)
            qr_lnbnn_data, _ = values
            for scale in qr_lnbnn_data:
                scale_set.add(scale)
            qr_lnbnn_data_list.append(qr_lnbnn_data)
        scale_list = sorted(list(scale_set))

    with ut.Timer('Loading database'):
        with ut.Timer('Checking database cache'):
            compute = force_cache_recompute

            index_filepath_dict = {}
            aids_filepath_dict = {}
            for scale in scale_list:
                args = (scale, ANNOT_INDEX_TREES, )
                base_directory = 'db_index_scale_%s_trees_%d' % args
                base_path = join(index_path, base_directory)

                index_filepath = join(base_path, 'index.ann')
                aids_filepath  = join(base_path, 'aids.pkl')

                index_filepath_dict[scale] = index_filepath
                aids_filepath_dict[scale]  = aids_filepath

                if not exists(index_filepath):
                    print('Missing: %r' % (index_filepath, ))
                    compute = True

                if not exists(aids_filepath):
                    print('Missing: %r' % (aids_filepath, ))
                    compute = True

            print('Compute indices = %r' % (compute, ))

        if compute:
            # Cache as a future job until it is complete, in case other threads are looking at this cache as well
            future_index_directory = '%s%s' % (FUTURE_PREFIX, index_directory, )
            future_index_path = join(cache_path, future_index_directory)
            ut.ensuredir(future_index_path)

            with ut.Timer('Loading database LNBNN descriptors from depc'):
                values = ibs.ibeis_plugin_curvrank_pipeline(aid_list=db_aid_list, config=config,
                                                            verbose=verbose, use_depc=use_depc,
                                                            use_depc_optimized=use_depc_optimized)
                db_lnbnn_data, _ = values

            with ut.Timer('Creating Annoy indices'):
                for scale in scale_list:
                    assert scale in db_lnbnn_data
                    index_filepath = index_filepath_dict[scale]
                    aids_filepath  = aids_filepath_dict[scale]

                    future_index_filepath = index_filepath.replace(index_path, future_index_path)
                    future_aids_filepath  = aids_filepath.replace(index_path,  future_index_path)

                    ut.ensuredir(split(future_index_filepath)[0])
                    ut.ensuredir(split(future_aids_filepath)[0])

                    if not exists(index_filepath):
                        print('Writing computed Annoy scale=%r index to %r...' % (scale, future_index_filepath, ))
                        descriptors, aids = db_lnbnn_data[scale]
                        F.build_lnbnn_index(descriptors, future_index_filepath, num_trees=ANNOT_INDEX_TREES)
                    else:
                        ut.copy(index_filepath, future_index_filepath)
                        print('Using existing Annoy scale=%r index in %r...' % (scale, index_filepath, ))

                    if not exists(aids_filepath):
                        print('Writing computed AIDs scale=%r to %r...' % (scale, future_aids_filepath, ))
                        ut.save_cPkl(future_aids_filepath, aids)
                        print('\t...saved')
                    else:
                        ut.copy(aids_filepath, future_aids_filepath)
                        print('Using existing AIDs scale=%r in %r...' % (scale, aids_filepath, ))

            with ut.Timer('Activating index by setting from future to live'):
                ut.delete(index_path)
                ut.move(future_index_path, index_path, verbose=True)

        with ut.Timer('Loading database AIDs from cache'):
            aids_dict = {}
            for scale in scale_list:
                aids_filepath = aids_filepath_dict[scale]
                assert exists(aids_filepath)
                aids_dict[scale] = ut.load_cPkl(aids_filepath)

    assert exists(index_path)

    with ut.Timer('Computing scores'):
        zipped = list(zip(qr_aids_list, qr_lnbnn_data_list))
        for qr_aid_list, qr_lnbnn_data in ut.ProgressIter(zipped, lbl='CurvRank Vectored Scoring', freq=1000):

            # Run LNBNN identification for each scale independently and aggregate
            score_dict = {}
            for scale in ut.ProgressIter(scale_list, lbl='Performing ANN inference', freq=1):
                assert scale in qr_lnbnn_data
                assert scale in index_filepath_dict
                assert scale in aids_dict

                qr_descriptors, _ = qr_lnbnn_data[scale]
                index_filepath    = index_filepath_dict[scale]

                assert exists(index_filepath)
                db_aids = aids_dict[scale]

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

            # Sparsify
            qr_aid_set = set(qr_aid_list)
            rowid_list = list(score_dict.keys())
            for rowid in rowid_list:
                score = score_dict[rowid]
                # Scores are non-positive floats (unless errored), delete scores that are 0.0 or positive.
                if score >= minimum_score or rowid in qr_aid_set:
                    score_dict.pop(rowid)

            yield qr_aid_list, score_dict


@register_ibs_method
def ibeis_plugin_curvrank(ibs, label, qaid_list, daid_list, config):
    r"""
    CommandLine:
        python -m ibeis_curvrank._plugin --exec-ibeis_plugin_curvrank

    Example:
        >>> # ENABLE_DOCTEST
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
        >>> # Call function normally
        >>> config = CurvRankDorsalConfig()
        >>> score_list = list(ibs.ibeis_plugin_curvrank('CurvRankTest', qaid_list, daid_list, config))
        >>> result = score_list
        >>> print(result)
        [(-0.0,), (1.9445927201886661,), (0.11260342702735215,), (0.06715983111644164,), (0.05171962268650532,), (0.08413137518800795,), (0.6862188717350364,), (1.0749932969920337,), (1.928582369175274,), (-0.0,), (0.3083178228698671,), (0.31571834394708276,), (0.144817239837721,), (0.4288492240011692,), (1.678820631466806,), (1.3525973158539273,), (0.31891411560354754,), (0.18176447856239974,), (-0.0,), (0.386130575905554,), (0.0972284316085279,), (0.19626294076442719,), (0.3404016795102507,), (0.16608526022173464,), (0.11954134894767776,), (0.2543876450508833,), (0.6982887189369649,), (-0.0,), (0.4541728966869414,), (0.30956776603125036,), (0.4229014730080962,), (0.22321902139810845,), (0.12588574923574924,), (0.09017095575109124,), (0.21655505849048495,), (0.5589789934456348,), (-0.0,), (6.011784115340561,), (0.4132015435025096,), (0.09880360751412809,), (0.19417243939824402,), (0.10126215778291225,), (0.24388620839454234,), (0.28090291377156973,), (5.304396523628384,), (-0.0,), (0.36655064788646996,), (0.18875156180001795,), (0.521016908576712,), (1.5610270453616977,), (0.31230442877858877,), (0.22889767913147807,), (0.1405167318880558,), (0.22574857133440673,), (-0.0,), (0.6370306296739727,), (1.092248206725344,), (2.110280451888684,), (0.08121629932429641,), (0.06134591973386705,), (0.10521706636063755,), (0.1293912068940699,), (0.762320066569373,), (-0.0,)]
    """
    print('Computing %s' % (label, ))

    cache_path = abspath(join(ibs.get_cachedir(), 'curvrank'))
    ut.ensuredir(cache_path)

    assert len(qaid_list) == len(daid_list), 'Lengths of qaid_list %d != daid_list %d' % (len(qaid_list), len(daid_list))

    qaid_list_  = sorted(list(set(qaid_list)))
    daid_list_  = sorted(list(set(daid_list)))

    qr_aids_list = [
        [qaid]
        for qaid in qaid_list_
    ]
    db_aid_list = daid_list_

    args = (label, len(qaid_list), len(qaid_list_), len(daid_list), len(daid_list_))
    message = 'Computing IBEIS CurvRank (%s) on %d total qaids (%d unique), %d total daids (%d unique)' % args
    with ut.Timer(message):
        value_iter = ibs.ibeis_plugin_curvrank_scores_depc(db_aid_list, qr_aids_list,
                                                           config=config,
                                                           use_names=False,
                                                           use_depc_optimized=USE_DEPC_OPTIMIZED)
        score_dict = {}
        for value in value_iter:
            qr_aid_list, score_dict_ = value
            assert len(qr_aid_list) == 1
            qaid = qr_aid_list[0]
            score_dict[qaid] = score_dict_

    zipped = list(zip(qaid_list, daid_list))
    for qaid, daid in ut.ProgressIter(zipped, 'CurvRank Pair-wise Final Scores', freq=1000):
        assert qaid in score_dict
        score = score_dict[qaid].get(daid, 0.0)
        score *= -1.0

        yield (score, )


@register_ibs_method
def ibeis_plugin_curvrank_delete_cache_optimized(ibs, aid_list, tablename):
    import networkx as nx

    assert tablename in ['CurvRankDorsal', 'CurvRankFluke', 'CurvRankFinfindrHybridDorsal']

    tablename_list = [
        'curvature_descriptor_optimized',
        tablename,
    ]

    graph = ibs.depc_annot.make_graph(implicit=True)
    root = ibs.depc_annot.root
    params_iter = list(zip(aid_list))

    for target_tablename in tablename_list:
        print(target_tablename)

        path = nx.shortest_path(graph, root, target_tablename)
        for parent, child in ut.itertwo(path):
            child_table = ibs.depc_annot[child]

            relevant_col_attrs = []
            for attrs in child_table.parent_col_attrs:
                if attrs['parent_table'] == parent:
                    relevant_col_attrs.append(attrs)

            parent_colnames = []
            for attrs in relevant_col_attrs:
                parent_colnames.append(attrs['intern_colname'])

            child_rowids = []
            for colname in parent_colnames:
                indexname = '%s_index' % (colname, )
                command = '''CREATE INDEX IF NOT EXISTS {indexname} ON {tablename} ({colname}, {rowid_colname});'''.format(
                    indexname=indexname,
                    tablename=child,
                    colname=colname,
                    rowid_colname=child_table.rowid_colname,
                )
                child_table.db.connection.execute(command).fetchall()

                child_rowids_ = child_table.db.get_where_eq_set(
                    child_table.tablename, (child_table.rowid_colname,),
                    params_iter, unpack_scalars=False,
                    where_colnames=[colname])
                # child_rowids_ = ut.flatten(child_rowids_)
                child_rowids += child_rowids_

            child_table.delete_rows(child_rowids, delete_extern=True)

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis_curvrank._plugin --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
