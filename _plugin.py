# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia.control import controller_inject  # NOQA
from os.path import abspath, join, exists, split
import wbia_curvrank.fcnn as fcnn
import wbia_curvrank.functional as F
import wbia_curvrank.regression as regression
from wbia_curvrank import imutils

# import wbia.constants as const
from scipy import interpolate
import numpy as np
import utool as ut
import vtool as vt
import datetime
import cv2
import torch

# We want to register the depc plugin functions as well, so import it here for IBEIS
import wbia_curvrank._plugin_depc  # NOQA
from wbia_curvrank._plugin_depc import (
    DEFAULT_SCALES,
    INDEX_NUM_TREES,
    INDEX_SEARCH_K,
    INDEX_LNBNN_K,
    INDEX_SEARCH_D,
    INDEX_NUM_ANNOTS,
    _convert_kwargs_config_to_depc_config,
)

(print, rrr, profile) = ut.inject2(__name__)

_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)
register_api = controller_inject.get_wbia_flask_api(__name__)


USE_DEPC = True
USE_DEPC_OPTIMIZED = True


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
        'localization': 'https://wildbookiarepository.azureedge.net/models/curvrank.localization.dorsal.weights.pkl',
        'segmentation': 'https://wildbookiarepository.azureedge.net/models/curvrank.segmentation.dorsal.weights.pkl',
    },
    'dorsalfinfindrhybrid': {
        'localization': 'https://wildbookiarepository.azureedge.net/models/curvrank.localization.dorsal.weights.pkl',
        'segmentation': 'https://wildbookiarepository.azureedge.net/models/curvrank.segmentation.dorsal.weights.pkl',
    },
    'fluke': {
        'localization': None,
        'segmentation': 'https://wildbookiarepository.azureedge.net/models/curvrank.segmentation.fluke.weights.pkl',
    },
}

if not HYBRID_FINFINDR_EXTRACTION_FAILURE_CURVRANK_FALLBACK:
    URL_DICT['dorsalfinfindrhybrid']['localization'] = None
    URL_DICT['dorsalfinfindrhybrid']['segmentation'] = None


@register_ibs_method
def wbia_plugin_curvrank_preprocessing(
    ibs, aid_list, pad=0.1, width_coarse=384, height_coarse=192, width_anchor=224, height_anchor=224, **kwargs
):
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
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_preprocessing
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_preprocessing:0
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_preprocessing:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> values = ibs.wbia_plugin_curvrank_preprocessing(aid_list)
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
        >>> from wbia_curvrank._plugin import *  # NOQA
        >>> from wbia_curvrank._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> model_type = 'fluke'
        >>> width = DEFAULT_WIDTH[model_type]
        >>> height = DEFAULT_HEIGHT[model_type]
        >>> values = ibs.wbia_plugin_curvrank_preprocessing(aid_list, width=width, height=height)
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
    gid_list = ibs.get_annot_gids(aid_list)
    image_list = ibs.get_images(gid_list)
    bboxes = ibs.get_annot_bboxes(aid_list)

    viewpoint_list = ibs.get_annot_viewpoints(aid_list)
    viewpoint_list = [
        None if viewpoint is None else viewpoint.lower() for viewpoint in viewpoint_list
    ]
    flip_list = [viewpoint in RIGHT_FLIP_LIST for viewpoint in viewpoint_list]
    pad_list = [pad] * len(aid_list)
    height_coarse_list = [height_coarse] * len(aid_list)
    width_coarse_list = [width_coarse] * len(aid_list)
    height_anchor_list = [height_anchor] * len(aid_list)
    width_anchor_list = [width_anchor] * len(aid_list)

    zipped = zip(image_list, bboxes, flip_list, pad_list, width_coarse_list, height_coarse_list, width_anchor_list, height_anchor_list)

    config_ = {
        'ordered': True,
        'chunksize': CHUNKSIZE,
        'force_serial': ibs.force_serial or FORCE_SERIAL,
        'progkw': {'freq': 10},
    }
    generator = ut.generate2(F.preprocess_image, zipped, nTasks=len(aid_list), **config_)

    resized_images_coarse, resized_images_anchor, cropped_images = [], [], []
    for coarse_resized_image, anchor_resized_image, cropped_image in generator:
        resized_images_coarse.append(resized_image_coarse)
        resized_images_anchor.append(resized_image_anchor)
        cropped_images.append(cropped_image)

    return resized_images_coarse, resized_images_anchor, cropped_images


@register_ibs_method
def wbia_plugin_curvrank_coarse_probabilities(ibs, resized_images, width_coarse=384, height_coarse=192, **kwargs):
    coarse_params = '/home/mankow/Research/CurvRankv2/data/cascadia/Training/weights/Jul01_16-57-40_rad-8.params'
    unet = fcnn.UNet()
    unet.load_state_dict(torch.load(coarse_params, map_location='cuda:0'))
    unet.cuda(None)
    unet.eval()
    coarse_probabilities = []
    for index, x in enumerate(resized_images):
        x = x.cuda(None)
        x = x.view(1, 3, height_coarse, width_coarse)
        with torch.no_grad():
            _, y_hat = unet(x)
        y_hat = y_hat.data.cpu().numpy().transpose(0, 2, 3, 1)
        probs = (255 * y_hat[0, :, :, 1]).astype(np.uint8)
        coarse_probabilities.append(probs)
    return coarse_probabilities


@register_ibs_method
def wbia_plugin_curvrank_fine_gradients(ibs, images):
    config_ = {
        'ordered': True,
        'chunksize': CHUNKSIZE,
        'force_serial': ibs.force_serial or FORCE_SERIAL,
        'progkw': {'freq': 10},
    }
    generator = ut.generate2(F.refine_by_gradient, zip(images), nTasks=len(images), **config_)

    fine_gradients = []
    for fine_gradient in generator:
        fine_gradients.append(fine_gradient)

    return fine_gradients


@register_ibs_method
def wbia_plugin_curvrank_anchor_points(ibs, original_images, anchor_images, width_fine=1152, width_anchor=224, height_anchor=224, **kwargs):
    anchor_params = '/home/mankow/Research/CurvRankv2/data/cascadia/Training/weights/Jun18_20-05-58_using-20th-pt.params'
    anchor_nn = regression.VGG16()
    anchor_nn.load_state_dict(torch.load(anchor_params))
    anchor_nn.cuda(None)
    anchor_nn.eval()
    anchor_points = []
    for index, x in enumerate(anchor_images):
        part_img = original_images[index]

        x = x.cuda(None)
        x = x.view(1, 3, height_anchor, width_anchor)
        with torch.no_grad():
            y0_hat, y1_hat = anchor_nn(x)
        y0_hat = y0_hat.data.cpu().numpy()
        y1_hat = y1_hat.data.cpu().numpy()

        ratio = width_fine / part_img.shape[1]
        part_img_resized = cv2.resize(
            part_img, (0, 0), fx=ratio, fy=ratio,
            interpolation=cv2.INTER_AREA)
        height, width = part_img_resized.shape[0:2]
        start = y0_hat * np.array([width, height])
        end = y1_hat * np.array([width, height])
        anchor_points.append({'start': start, 'end': end})

    return anchor_points


@register_ibs_method
def wbia_plugin_curvrank_contours(ibs, original_images, coarse_probabilities, fine_gradients, endpoints, trim=0, width_fine=1152, **kwargs):
    trim_list = [trim] * len(original_images)
    width_fine_list = [width_fine] * len(original_images)

    zipped = zip(original_images, coarse_probabilities, fine_gradients, endpoints, trim_list, width_fine_list)

    config_ = {
        'ordered': True,
        'chunksize': CHUNKSIZE,
        'force_serial': ibs.force_serial or FORCE_SERIAL,
        'progkw': {'freq': 10},
    }
    generator = ut.generate2(F.contour_from_anchorpoints, zipped, nTasks=len(original_images), **config_)

    contours = []
    for contour in generator:
        contours.append(contour)

    return contours


@register_ibs_method
def wbia_plugin_curvrank_curvatures(ibs, contours, width_fine=1152, height_fine=576, scales=DEFAULT_SCALES['fluke'], transpose_dims=True, **kwargs):
    height_fine_list = [height_fine] * len(contours)
    width_fine_list = [width_fine] * len(contours)
    scales_list = [scales] * len(contours)
    transpose_dims_list = [transpose_dims] * len(contours)

    zipped = zip(contours, width_fine_list, height_fine_list, scales_list, transpose_dims_list)

    config_ = {
        'ordered': True,
        'chunksize': CHUNKSIZE,
        'force_serial': ibs.force_serial or FORCE_SERIAL,
        'progkw': {'freq': 10},
    }
    generator = ut.generate2(F.curvature, zipped, nTasks=len(contours), **config_)

    curvatures = []
    for curvature in generator:
        curvatures.append(curvature)

    return curvatures


@register_ibs_method
def wbia_plugin_curvrank_descriptors(ibs, contours, curvatures, scales=DEFAULT_SCALES['fluke'], curv_length=1024, feat_dim=32, num_keypoints=32, **kwargs):
    scales_list = [scales] * len(contours)
    curv_length_list = [curv_length] * len(contours)
    feat_dim_list = [feat_dim] * len(contours)
    num_keypoints_list = [num_keypoints] * len(contours)

    zipped = zip(contours, curvatures, scales_list, curv_length_list, feat_dim_list, num_keypoints_list)

    config_ = {
        'ordered': True,
        'chunksize': CHUNKSIZE,
        'force_serial': ibs.force_serial or FORCE_SERIAL,
        'progkw': {'freq': 10},
    }
    generator = ut.generate2(F.curvature_descriptors, zipped)

    descriptors = []
    for descriptor in generator:
        descriptors.append(descriptor)

    return descriptors


@register_ibs_method
def wbia_plugin_curvrank_pipeline_compute(ibs, aid_list, config={}):
    r"""
    Args:
        ibs       (IBEISController): IBEIS controller object
        success_list: output of wbia_plugin_curvrank_outline
        outlines (list of np.ndarray): output of wbia_plugin_curvrank_outline

    Returns:
        success_list_
        curvature_descriptors

    CommandLine:
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_pipeline_compute
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_pipeline_compute:0
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_pipeline_compute:1
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_pipeline_compute:2
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_pipeline_compute:3

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> values = ibs.wbia_plugin_curvrank_pipeline_compute(aid_list)
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
        >>> from wbia_curvrank._plugin import *  # NOQA
        >>> from wbia_curvrank._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
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
        >>> values = ibs.wbia_plugin_curvrank_pipeline_compute(aid_list, config=config)
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
        >>> from wbia_curvrank._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> aid_list *= 20
        >>> values = ibs.wbia_plugin_curvrank_pipeline_compute(aid_list)
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
        >>> from wbia_curvrank._plugin import *  # NOQA
        >>> from wbia_curvrank._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
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
        >>> values = ibs.wbia_plugin_curvrank_pipeline_compute(aid_list, config=config)
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
    values = ibs.wbia_plugin_curvrank_preprocessing(aid_list, **config)
    coarse_resized_images, anchor_resized_images, original_images = values

    values = ibs.wbia_plugin_curvrank_coarse_probabilities(coarse_resized_images, **config)
    coarse_probabilities = values

    values = ibs.wbia_plugin_curvrank_fine_gradients(original_images)
    fine_gradients = values

    values = ibs.wbia_plugin_curvrank_anchor_points(original_images, anchor_resized_images, **config)
    endpoints = values

    values = ibs.wbia_plugin_curvrank_contours(original_images, coarse_probabilities, fine_gradients, endpoints, **config)
    contours = values

    values = ibs.wbia_plugin_curvrank_curvatures(contours, **config)
    curvatures = values

    values = ibs.wbia_plugin_curvrank_descriptors(contours, curvatures, **config)
    descriptors = values

    success = [True for d in descriptors]

    return success, descriptors


@register_ibs_method
def wbia_plugin_curvrank_pipeline_aggregate(
    ibs, aid_list, success_list, descriptor_dict_list
):
    r"""
    Args:
        ibs       (IBEISController): IBEIS controller object
        success_list: output of wbia_plugin_curvrank_outline
        outlines (list of np.ndarray): output of wbia_plugin_curvrank_outline

    Returns:
        success_list_
        curvature_descriptors

    CommandLine:
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_pipeline_aggregate
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_pipeline_aggregate:0
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_pipeline_aggregate:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> values = ibs.wbia_plugin_curvrank_pipeline_compute(aid_list)
        >>> success_list, curvature_descriptor_dicts = values
        >>> lnbnn_dict = ibs.wbia_plugin_curvrank_pipeline_aggregate(aid_list, success_list, curvature_descriptor_dicts)
        >>> hash_list = [
        >>>     ut.hash_data(lnbnn_dict[scale])
        >>>     for scale in sorted(list(lnbnn_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['wntkcfldlkhsxvnzvthlenvjrcxblmtd']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank._plugin import *  # NOQA
        >>> from wbia_curvrank._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
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
        >>> values = ibs.wbia_plugin_curvrank_pipeline_compute(aid_list, config=config)
        >>> success_list, curvature_descriptor_dicts = values
        >>> lnbnn_dict = ibs.wbia_plugin_curvrank_pipeline_aggregate(aid_list, success_list, curvature_descriptor_dicts)
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

        descriptor_dict = descriptor_dict['descriptors'] #TODO: Maybe change this?
        for scale in descriptor_dict:
            if scale not in lnbnn_dict:
                lnbnn_dict[scale] = {
                    'descriptors': [],
                    'aids': [],
                }

            descriptors = descriptor_dict[scale]
            aids = [aid] * descriptors.shape[0]

            lnbnn_dict[scale]['descriptors'].append(descriptors)
            lnbnn_dict[scale]['aids'].append(aids)

    for scale in lnbnn_dict:
        descriptors = np.vstack(lnbnn_dict[scale]['descriptors'])
        assert np.allclose(
            np.linalg.norm(descriptors, axis=1), np.ones(descriptors.shape[0])
        )

        aids = np.hstack(lnbnn_dict[scale]['aids'])
        lnbnn_dict[scale] = (
            descriptors,
            aids,
        )

    return lnbnn_dict


@register_ibs_method
def wbia_plugin_curvrank_pipeline(
    ibs,
    imageset_rowid=None,
    aid_list=None,
    config={},
    use_depc=USE_DEPC,
    use_depc_optimized=USE_DEPC_OPTIMIZED,
    verbose=False,
):
    r"""
    Args:
        ibs       (IBEISController): IBEIS controller object
        success_list: output of wbia_plugin_curvrank_outline
        outlines (list of np.ndarray): output of wbia_plugin_curvrank_outline

    Returns:
        success_list_
        curvature_descriptors

    CommandLine:
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_pipeline
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_pipeline:0
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_pipeline:1
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_pipeline:2
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_pipeline:3
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_pipeline:4
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_pipeline:5

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> lnbnn_dict, aid_list = ibs.wbia_plugin_curvrank_pipeline(aid_list=aid_list, use_depc=False)
        >>> hash_list = [
        >>>     ut.hash_data(lnbnn_dict[scale])
        >>>     for scale in sorted(list(lnbnn_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['wntkcfldlkhsxvnzvthlenvjrcxblmtd']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> lnbnn_dict, aid_list = ibs.wbia_plugin_curvrank_pipeline(aid_list=aid_list, use_depc=True, use_depc_optimized=False)
        >>> hash_list = [
        >>>     ut.hash_data(lnbnn_dict[scale])
        >>>     for scale in sorted(list(lnbnn_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['wntkcfldlkhsxvnzvthlenvjrcxblmtd']

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> lnbnn_dict, aid_list = ibs.wbia_plugin_curvrank_pipeline(aid_list=aid_list, use_depc=True, use_depc_optimized=True)
        >>> hash_list = [
        >>>     ut.hash_data(lnbnn_dict[scale])
        >>>     for scale in sorted(list(lnbnn_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['wntkcfldlkhsxvnzvthlenvjrcxblmtd']

    Example3:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank._plugin import *  # NOQA
        >>> from wbia_curvrank._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
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
        >>> lnbnn_dict, aid_list = ibs.wbia_plugin_curvrank_pipeline(aid_list=aid_list, config=config, use_depc=False)
        >>> hash_list = [
        >>>     ut.hash_data(lnbnn_dict[scale])
        >>>     for scale in sorted(list(lnbnn_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['ylxevmyeygxlhbcsuwzakfnlisbantdr']

    Example4:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank._plugin import *  # NOQA
        >>> from wbia_curvrank._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
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
        >>> lnbnn_dict, aid_list = ibs.wbia_plugin_curvrank_pipeline(aid_list=aid_list, config=config, use_depc=True, use_depc_optimized=False)
        >>> hash_list = [
        >>>     ut.hash_data(lnbnn_dict[scale])
        >>>     for scale in sorted(list(lnbnn_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['ylxevmyeygxlhbcsuwzakfnlisbantdr']

    Example5:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank._plugin import *  # NOQA
        >>> from wbia_curvrank._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
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
        >>> lnbnn_dict, aid_list = ibs.wbia_plugin_curvrank_pipeline(aid_list=aid_list, config=config, use_depc=True, use_depc_optimized=True)
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
        table_name = (
            'curvature_descriptor_optimized'
            if use_depc_optimized
            else 'curvature_descriptor'
        )
        success_list = ibs.depc_annot.get(table_name, aid_list, 'success', config=config_)
        descriptor_dict_list = ibs.depc_annot.get(
            table_name, aid_list, 'descriptor', config=config_
        )
    else:
        values = ibs.wbia_plugin_curvrank_pipeline_compute(aid_list, config=config)
        success_list, descriptor_dict_list = values

    if verbose:
        print('\tAggregate Pipeline Results')

    lnbnn_dict = ibs.wbia_plugin_curvrank_pipeline_aggregate(
        aid_list, success_list, descriptor_dict_list
    )

    return lnbnn_dict, aid_list


@register_ibs_method
def wbia_plugin_curvrank_scores(
    ibs,
    db_aid_list,
    qr_aids_list,
    config={},
    verbose=False,
    use_names=True,
    minimum_score=-1e-5,
    use_depc=USE_DEPC,
    use_depc_optimized=USE_DEPC_OPTIMIZED,
):
    r"""
    CurvRank Example

    Args:
        ibs       (IBEISController): IBEIS controller object
        lnbnn_k   (int): list of image rowids (aids)

    Returns:
        score_dict

    CommandLine:
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_scores
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_scores:0
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_scores:1
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_scores:2
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_scores:3
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_scores:4
        python -m wbia_curvrank._plugin --test-wbia_plugin_curvrank_scores:5

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> db_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Dorsal Database')
        >>> db_aid_list = ibs.get_imageset_aids(db_imageset_rowid)
        >>> qr_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Dorsal Query')
        >>> qr_aid_list = ibs.get_imageset_aids(qr_imageset_rowid)
        >>> score_dict_iter = ibs.wbia_plugin_curvrank_scores(db_aid_list, [qr_aid_list], use_depc=False)
        >>> score_dict_list = list(score_dict_iter)
        >>> qr_aid_list, score_dict = score_dict_list[0]
        >>> for key in score_dict:
        >>>     score_dict[key] = round(score_dict[key], 8)
        >>> result = score_dict
        >>> print(result)
        {1: -31.81339289, 2: -3.7092349, 3: -4.95274189}

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> db_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Dorsal Database')
        >>> db_aid_list = ibs.get_imageset_aids(db_imageset_rowid)
        >>> qr_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Dorsal Query')
        >>> qr_aid_list = ibs.get_imageset_aids(qr_imageset_rowid)
        >>> score_dict_iter = ibs.wbia_plugin_curvrank_scores(db_aid_list, [qr_aid_list], use_depc=True, use_depc_optimized=False)
        >>> score_dict_list = list(score_dict_iter)
        >>> qr_aid_list, score_dict = score_dict_list[0]
        >>> for key in score_dict:
        >>>     score_dict[key] = round(score_dict[key], 8)
        >>> result = score_dict
        >>> print(result)
        {1: -31.81339289, 2: -3.7092349, 3: -4.95274189}

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> db_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Dorsal Database')
        >>> db_aid_list = ibs.get_imageset_aids(db_imageset_rowid)
        >>> qr_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Dorsal Query')
        >>> qr_aid_list = ibs.get_imageset_aids(qr_imageset_rowid)
        >>> score_dict_iter = ibs.wbia_plugin_curvrank_scores(db_aid_list, [qr_aid_list], use_depc=True, use_depc_optimized=True)
        >>> score_dict_list = list(score_dict_iter)
        >>> qr_aid_list, score_dict = score_dict_list[0]
        >>> for key in score_dict:
        >>>     score_dict[key] = round(score_dict[key], 8)
        >>> result = score_dict
        >>> print(result)
        {1: -31.81339289, 2: -3.7092349, 3: -4.95274189}

    Example3:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank._plugin import *  # NOQA
        >>> from wbia_curvrank._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
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
        >>> score_dict_iter = ibs.wbia_plugin_curvrank_scores(db_aid_list, [qr_aid_list], config=config, use_depc=False)
        >>> score_dict_list = list(score_dict_iter)
        >>> qr_aid_list, score_dict = score_dict_list[0]
        >>> for key in score_dict:
        >>>     score_dict[key] = round(score_dict[key], 8)
        >>> result = score_dict
        >>> print(result)
        {14: -1.00862974, 7: -0.55433992, 8: -0.70058628, 9: -0.3044969, 10: -0.27739539, 11: -7.8684881, 12: -1.01431028, 13: -1.46861451}

    Example4:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank._plugin import *  # NOQA
        >>> from wbia_curvrank._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
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
        >>> score_dict_iter = ibs.wbia_plugin_curvrank_scores(db_aid_list, [qr_aid_list], config=config, use_depc=True, use_depc_optimized=False)
        >>> score_dict_list = list(score_dict_iter)
        >>> qr_aid_list, score_dict = score_dict_list[0]
        >>> for key in score_dict:
        >>>     score_dict[key] = round(score_dict[key], 8)
        >>> result = score_dict
        >>> print(result)
        {14: -1.00862974, 7: -0.55433992, 8: -0.70058628, 9: -0.3044969, 10: -0.27739539, 11: -7.8684881, 12: -1.01431028, 13: -1.46861451}

    Example5:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank._plugin import *  # NOQA
        >>> from wbia_curvrank._plugin_depc import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
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
        >>> score_dict_iter = ibs.wbia_plugin_curvrank_scores(db_aid_list, [qr_aid_list], config=config, use_depc=True, use_depc_optimized=True)
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
    TTL_HOUR_DELETE = 7 * 24
    TTL_HOUR_PREVIOUS = 2 * 24

    use_daily_cache = config.pop('use_daily_cache', False)
    daily_cache_tag = config.pop('daily_cache_tag', 'global')
    force_cache_recompute = config.pop('force_cache_recompute', False)

    num_trees = config.pop('num_trees', INDEX_NUM_TREES)
    search_k = config.pop('search_k', INDEX_SEARCH_K)
    lnbnn_k = config.pop('lnbnn_k', INDEX_LNBNN_K)

    args = (
        use_daily_cache,
        daily_cache_tag,
        force_cache_recompute,
    )
    print(
        'CurvRank cache config:\n\tuse_daily_cache = %r\n\tdaily_cache_tag = %r\n\tforce_cache_recompute = %r\n\t'
        % args
    )
    print('CurvRank num_trees   : %r' % (num_trees,))
    print('CurvRank search_k    : %r' % (search_k,))
    print('CurvRank lnbnn_k     : %r' % (lnbnn_k,))
    print('CurvRank algo config : %s' % (ut.repr3(config),))

    config_hash = ut.hash_data(ut.repr3(config))
    now = datetime.datetime.now()
    timestamp_fmtstr = '%Y-%m-%d-%H-%M-%S'
    timestamp = now.strftime(timestamp_fmtstr)

    daily_cache_tag = str(daily_cache_tag)
    if daily_cache_tag in [None, '']:
        daily_cache_tag = 'global'

    if daily_cache_tag in ['global']:
        qr_aid_list = ut.flatten(qr_aids_list)
        qr_species_set = set(ibs.get_annot_species_texts(qr_aid_list))
        qr_species_str = '-'.join(sorted(qr_species_set))
        daily_index_hash = 'daily-global-%s' % (qr_species_str)
    else:
        daily_index_hash = 'daily-tag-%s' % (daily_cache_tag)

    with ut.Timer('Clearing old caches (TTL = %d hours)' % (TTL_HOUR_DELETE,)):

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
                print('Checking %r (%r)' % (directory, then,))

                if then < past_delete:
                    print('\ttoo old, deleting %r...' % (path,))
                    ut.delete(path)
                else:
                    if past_previous <= then:
                        daily_index_search_str = '_hash_%s_config_' % (daily_index_hash,)
                        if daily_index_search_str in directory:
                            available_previous_list.append(directory)
                    delta = then - past_delete
                    hours = delta.total_seconds() / 60 / 60
                    print('\tkeeping cache for %0.2f more hours...' % (hours,))
            except Exception:
                print('\tinvalid (parse error), deleting %r...' % (path,))
                ut.delete(path)

        # Check for any FUTURE_PREFIX folders that are too old (due to an error) and need to be deleted
        for path in ut.glob(join(cache_path, '%sindex_*' % (FUTURE_PREFIX,))):
            try:
                directory = split(path)[1]
                directory = directory.replace(FUTURE_PREFIX, '')
                date_str = directory.split('_')[1]
                then = datetime.datetime.strptime(date_str, timestamp_fmtstr)
                print('Checking %r (%r)' % (directory, then,))

                if then < past_delete:
                    print('\ttoo old, deleting %r...' % (path,))
                    ut.delete(path)
            except Exception:
                print('\tinvalid (parse error), deleting %r...' % (path,))
                ut.delete(path)

        available_previous_list = sorted(available_previous_list)
        args = (ut.repr3(available_previous_list),)
        print('\nAvailable previous cached: %s' % args)

    all_aid_list = ut.flatten(qr_aids_list) + db_aid_list

    if use_daily_cache:
        if force_cache_recompute or len(available_previous_list) == 0:
            args = (
                timestamp,
                daily_index_hash,
                config_hash,
            )
            index_directory = 'index_%s_hash_%s_config_%s' % args
            print(
                'Using daily index (recompute = %r): %r'
                % (force_cache_recompute, index_directory,)
            )
        else:
            index_directory = available_previous_list[-1]
            print('Using the most recent available index: %r' % (index_directory,))
    else:
        all_annot_uuid_list = ibs.get_annot_uuids(sorted(all_aid_list))
        index_hash = ut.hash_data(all_annot_uuid_list)

        args = (
            timestamp,
            index_hash,
            config_hash,
        )
        index_directory = 'index_%s_hash_%s_config_%s' % args
        print('Using hashed index: %r' % (index_directory,))

    if daily_cache_tag in ['global']:
        num_annots = len(all_aid_list)
        num_trees_ = int(np.ceil(num_annots / INDEX_NUM_ANNOTS))
        num_trees_ = max(num_trees, num_trees_)
        search_k_ = lnbnn_k * num_trees_ * INDEX_SEARCH_D
        if num_trees_ != num_trees:
            print(
                '[global] WARNING! Using num_trees = %d instead of %d (based on %d annotations)'
                % (num_trees_, num_trees, num_annots,)
            )
            print(
                '[global] WARNING! Using search_k = %d instead of %d (based on %d annotations)'
                % (search_k_, search_k, num_annots,)
            )
            num_trees = num_trees_
            search_k = search_k_

    index_path = join(cache_path, index_directory)

    with ut.Timer('Loading query'):
        scale_set = set([])
        qr_lnbnn_data_list = []
        for qr_aid_list in ut.ProgressIter(
            qr_aids_list, lbl='CurvRank Query LNBNN', freq=1000
        ):
            values = ibs.wbia_plugin_curvrank_pipeline(
                aid_list=qr_aid_list,
                config=config,
                verbose=verbose,
                use_depc=use_depc,
                use_depc_optimized=use_depc_optimized,
            )
            qr_lnbnn_data, _ = values
            for scale in qr_lnbnn_data:
                scale_set.add(scale)
            qr_lnbnn_data_list.append(qr_lnbnn_data)
        scale_list = sorted(list(scale_set))

    if not exists(index_path):
        force_cache_recompute = True

    with ut.Timer('Loading database'):
        with ut.Timer('Checking database cache'):
            compute = force_cache_recompute

            index_filepath_dict = {}
            aids_filepath_dict = {}
            for scale in scale_list:
                base_directory_fmtstr = 'db_index_scale_%s_trees_%s'

                args = (scale, '*')
                base_directory = base_directory_fmtstr % args
                base_path = join(index_path, base_directory)

                base_path_list = ut.glob(base_path)
                if daily_cache_tag in ['global'] and len(base_path_list) == 1:
                    base_path = base_path_list[0]

                    try:
                        num_trees_ = int(base_path.strip().strip('/').split('_')[1])
                        search_k_ = lnbnn_k * num_trees_ * INDEX_SEARCH_D
                        if num_trees_ != num_trees:
                            print(
                                '[local] WARNING! Using num_trees = %d instead of %d (based on %d annotations)'
                                % (num_trees_, num_trees, num_annots,)
                            )
                            print(
                                '[local] WARNING! Using search_k = %d instead of %d (based on %d annotations)'
                                % (search_k_, search_k, num_annots,)
                            )
                            num_trees = num_trees_
                            search_k = search_k_
                    except Exception:
                        pass
                else:
                    args = (
                        scale,
                        num_trees,
                    )
                    base_directory = base_directory_fmtstr % args
                    base_path = join(index_path, base_directory)

                if not exists(index_path):
                    print('Missing: %r' % (index_path,))
                    compute = True

                if not exists(base_path):
                    print('Missing: %r' % (base_path,))
                    compute = True

                index_filepath = join(base_path, 'index.ann')
                aids_filepath = join(base_path, 'aids.pkl')

                index_filepath_dict[scale] = index_filepath
                aids_filepath_dict[scale] = aids_filepath

                if not exists(index_filepath):
                    print('Missing: %r' % (index_filepath,))
                    compute = True

                if not exists(aids_filepath):
                    print('Missing: %r' % (aids_filepath,))
                    compute = True

            print('Compute indices = %r' % (compute,))

        if compute:
            # Cache as a future job until it is complete, in case other threads are looking at this cache as well
            future_index_directory = '%s%s' % (FUTURE_PREFIX, index_directory,)
            future_index_path = join(cache_path, future_index_directory)
            ut.ensuredir(future_index_path)

            with ut.Timer('Loading database LNBNN descriptors from depc'):
                values = ibs.wbia_plugin_curvrank_pipeline(
                    aid_list=db_aid_list,
                    config=config,
                    verbose=verbose,
                    use_depc=use_depc,
                    use_depc_optimized=use_depc_optimized,
                )
                db_lnbnn_data, _ = values

            with ut.Timer('Creating Annoy indices'):
                for scale in scale_list:
                    assert scale in db_lnbnn_data
                    index_filepath = index_filepath_dict[scale]
                    aids_filepath = aids_filepath_dict[scale]

                    future_index_filepath = index_filepath.replace(
                        index_path, future_index_path
                    )
                    future_aids_filepath = aids_filepath.replace(
                        index_path, future_index_path
                    )

                    ut.ensuredir(split(future_index_filepath)[0])
                    ut.ensuredir(split(future_aids_filepath)[0])

                    if not exists(index_filepath):
                        print(
                            'Writing computed Annoy scale=%r index to %r...'
                            % (scale, future_index_filepath,)
                        )
                        descriptors, aids = db_lnbnn_data[scale]
                        F.build_lnbnn_index(
                            descriptors, future_index_filepath, num_trees=num_trees
                        )
                    else:
                        ut.copy(index_filepath, future_index_filepath)
                        print(
                            'Using existing Annoy scale=%r index in %r...'
                            % (scale, index_filepath,)
                        )

                    if not exists(aids_filepath):
                        print(
                            'Writing computed AIDs scale=%r to %r...'
                            % (scale, future_aids_filepath,)
                        )
                        ut.save_cPkl(future_aids_filepath, aids)
                        print('\t...saved')
                    else:
                        ut.copy(aids_filepath, future_aids_filepath)
                        print(
                            'Using existing AIDs scale=%r in %r...'
                            % (scale, aids_filepath,)
                        )

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
        for qr_aid_list, qr_lnbnn_data in ut.ProgressIter(
            zipped, lbl='CurvRank Vectored Scoring', freq=1000
        ):

            # Run LNBNN identification for each scale independently and aggregate
            score_dict = {}
            for scale in ut.ProgressIter(
                scale_list, lbl='Performing ANN inference', freq=1
            ):
                assert scale in qr_lnbnn_data
                assert scale in index_filepath_dict
                assert scale in aids_dict

                qr_descriptors, _ = qr_lnbnn_data[scale]
                index_filepath = index_filepath_dict[scale]

                assert exists(index_filepath)
                db_aids = aids_dict[scale]

                if use_names:
                    db_rowids = ibs.get_annot_nids(db_aids)
                else:
                    db_rowids = db_aids

                score_dict_ = F.lnbnn_identify(
                    index_filepath, lnbnn_k, qr_descriptors, db_rowids, search_k=search_k
                )
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
def wbia_plugin_curvrank(ibs, label, qaid_list, daid_list, config):
    r"""
    CommandLine:
        python -m wbia_curvrank._plugin --exec-wbia_plugin_curvrank

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank._plugin_depc import *  # NOQA
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
        >>> # Call function normally
        >>> config = CurvRankDorsalConfig()
        >>> score_list = list(ibs.wbia_plugin_curvrank('CurvRankTest', qaid_list, daid_list, config))
        >>> result = score_list
        >>> print(result)
        [(-0.0,), (1.9445927201886661,), (0.11260342702735215,), (0.06715983111644164,), (0.05171962268650532,), (0.08413137518800795,), (0.6862188717350364,), (1.0749932969920337,), (1.928582369175274,), (-0.0,), (0.3083178228698671,), (0.31571834394708276,), (0.144817239837721,), (0.4288492240011692,), (1.678820631466806,), (1.3525973158539273,), (0.31891411560354754,), (0.18176447856239974,), (-0.0,), (0.386130575905554,), (0.0972284316085279,), (0.19626294076442719,), (0.3404016795102507,), (0.16608526022173464,), (0.11954134894767776,), (0.2543876450508833,), (0.6982887189369649,), (-0.0,), (0.4541728966869414,), (0.30956776603125036,), (0.4229014730080962,), (0.22321902139810845,), (0.12588574923574924,), (0.09017095575109124,), (0.21655505849048495,), (0.5589789934456348,), (-0.0,), (6.011784115340561,), (0.4132015435025096,), (0.09880360751412809,), (0.19417243939824402,), (0.10126215778291225,), (0.24388620839454234,), (0.28090291377156973,), (5.304396523628384,), (-0.0,), (0.36655064788646996,), (0.18875156180001795,), (0.521016908576712,), (1.5610270453616977,), (0.31230442877858877,), (0.22889767913147807,), (0.1405167318880558,), (0.22574857133440673,), (-0.0,), (0.6370306296739727,), (1.092248206725344,), (2.110280451888684,), (0.08121629932429641,), (0.06134591973386705,), (0.10521706636063755,), (0.1293912068940699,), (0.762320066569373,), (-0.0,)]
    """
    print('Computing %s' % (label,))

    cache_path = abspath(join(ibs.get_cachedir(), 'curvrank'))
    ut.ensuredir(cache_path)

    assert len(qaid_list) == len(daid_list), 'Lengths of qaid_list %d != daid_list %d' % (
        len(qaid_list),
        len(daid_list),
    )

    qaid_list_ = sorted(list(set(qaid_list)))
    daid_list_ = sorted(list(set(daid_list)))

    qr_aids_list = [[qaid] for qaid in qaid_list_]
    db_aid_list = daid_list_

    args = (label, len(qaid_list), len(qaid_list_), len(daid_list), len(daid_list_))
    message = (
        'Computing IBEIS CurvRank (%s) on %d total qaids (%d unique), %d total daids (%d unique)'
        % args
    )
    with ut.Timer(message):
        value_iter = ibs.wbia_plugin_curvrank_scores_depc(
            db_aid_list,
            qr_aids_list,
            config=config,
            use_names=False,
            use_depc_optimized=USE_DEPC_OPTIMIZED,
        )
        score_dict = {}
        for value in value_iter:
            qr_aid_list, score_dict_ = value
            assert len(qr_aid_list) == 1
            qaid = qr_aid_list[0]
            score_dict[qaid] = score_dict_

    zipped = list(zip(qaid_list, daid_list))
    for qaid, daid in ut.ProgressIter(
        zipped, 'CurvRank Pair-wise Final Scores', freq=1000
    ):
        assert qaid in score_dict
        score = score_dict[qaid].get(daid, 0.0)
        score *= -1.0

        yield (score,)


@register_ibs_method
def wbia_plugin_curvrank_delete_cache_optimized(ibs, aid_list, tablename):
    import networkx as nx

    assert tablename in [
        'CurvRankDorsal',
        'CurvRankFluke',
        'CurvRankFinfindrHybridDorsal',
    ]

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
                indexname = '%s_index' % (colname,)
                command = """CREATE INDEX IF NOT EXISTS {indexname} ON {tablename} ({colname}, {rowid_colname});""".format(
                    indexname=indexname,
                    tablename=child,
                    colname=colname,
                    rowid_colname=child_table.rowid_colname,
                )
                child_table.db.connection.execute(command).fetchall()

                child_rowids_ = child_table.db.get_where_eq_set(
                    child_table.tablename,
                    (child_table.rowid_colname,),
                    params_iter,
                    unpack_scalars=False,
                    where_colnames=[colname],
                )
                # child_rowids_ = ut.flatten(child_rowids_)
                child_rowids += child_rowids_

            child_table.delete_rows(child_rowids, delete_extern=True)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia_curvrank._plugin --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
