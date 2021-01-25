# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia.control import controller_inject  # NOQA
from os.path import abspath, join, exists, split
import wbia_curvrank_v2.fcnn as fcnn
import wbia_curvrank_v2.functional as F
import wbia_curvrank_v2.regression as regression
from wbia_curvrank_v2 import algo

# import wbia.constants as const
import numpy as np
import utool as ut
import datetime
import cv2
import torch

# We want to register the depc plugin functions as well, so import it here for IBEIS
import wbia_curvrank_v2._plugin_depc  # NOQA
from wbia_curvrank_v2._plugin_depc import (
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
# FORCE_SERIAL = FORCE_SERIAL or 'macosx' in ut.get_plat_specifier().lower()
# FORCE_SERIAL = FORCE_SERIAL or const.CONTAINERIZED
CHUNKSIZE = 16


RIGHT_FLIP_LIST = [  # CASE IN-SENSITIVE
    'right',
    'r',
    'dorsal fin right',
    'dorsal_fin_right',
]


MODEL_URL_DICT = {
    'anchor.dorsal': 'https://wildbookiarepository.azureedge.net/models/curvrank.v2.anchor.bottlenose.dorsal.params',
    'anchor.ear': 'https://wildbookiarepository.azureedge.net/models/curvrank.v2.anchor.elephant.ear.params',
    'anchor.fluke': 'https://wildbookiarepository.azureedge.net/models/curvrank.v2.anchor.humpback.fluke.params',
    'coarse.dorsal': 'https://wildbookiarepository.azureedge.net/models/curvrank.v2.coarse.bottlenose.dorsal.params',
    'coarse.ear': 'https://wildbookiarepository.azureedge.net/models/curvrank.v2.coarse.elephant.ear.params',
    'coarse.fluke': 'https://wildbookiarepository.azureedge.net/models/curvrank.v2.coarse.humpback.fluke.params',
    'fine.dorsal': 'https://wildbookiarepository.azureedge.net/models/curvrank.v2.fine.fcnn.params.chkpt',
    'fine.ear': 'https://wildbookiarepository.azureedge.net/models/curvrank.v2.fine.elephant.ear.params',
    'fine.fluke': 'https://wildbookiarepository.azureedge.net/models/curvrank.v2.fine.humpback.fluke.params.chkpt',
    'fine.fcnn': 'https://wildbookiarepository.azureedge.net/models/curvrank.v2.fine.fcnn.params.chkpt',
}


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@register_ibs_method
def wbia_plugin_curvrank_v2_preprocessing(ibs, aid_list, pad=0.1, **kwargs):
    r"""
    Pre-process images for CurvRank

    Args:
        ibs       (IBEISController): IBEIS controller object
        aid_list  (list of int): list of annotation rowids (aids)
        pad       (float in (0,1)): fraction of image with to pad

    Returns:
        cropped_images

    CommandLine:
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_preprocessing
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_preprocessing:0
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_preprocessing:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> _, cropped_images, _ = ibs.wbia_plugin_curvrank_v2_preprocessing(aid_list)
        >>> cropped_image = cropped_images[0]
        >>> assert ut.hash_data(cropped_image) in ['zrtghjovbhnangjdlsqtfvrntlzqmaey']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> aid_list *= 10
        >>> _, cropped_images, _ = ibs.wbia_plugin_curvrank_v2_preprocessing(aid_list)
        >>> cropped_image = cropped_images[0]
        >>> assert ut.hash_data(cropped_image) in ['zrtghjovbhnangjdlsqtfvrntlzqmaey']
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

    zipped = zip(image_list, bboxes, flip_list, pad_list)

    config_ = {
        'ordered': True,
        'chunksize': CHUNKSIZE,
        'force_serial': False,
        'progkw': {'freq': 10},
    }
    generator = ut.generate2(F.preprocess_image, zipped, nTasks=len(aid_list), **config_)

    images = []
    cropped_images = []
    cropped_bboxes = []
    for img, cropped_image, cropped_bbox in generator:
        images.append(img)
        cropped_images.append(cropped_image)
        cropped_bboxes.append(cropped_bbox)

    return images, cropped_images, cropped_bboxes


@register_ibs_method
def wbia_plugin_curvrank_v2_coarse_probabilities(
    ibs, cropped_images, width_coarse=384, height_coarse=192, model_type='fluke', **kwargs
):
    r"""
    Extract coarse probabilities for CurvRank

    Args:
        ibs             (IBEISController): IBEIS controller object
        cropped_images  (list of np.ndarray): BGR images
        width_coarse    (int): width of output
        height_coarse   (int): height of output

    Returns:
        coarse_probabilities

    CommandLine:
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_coarse_probabilities
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_coarse_probabilities:0
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_coarse_probabilities:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> _, cropped_images, _ = ibs.wbia_plugin_curvrank_v2_preprocessing(aid_list)
        >>> coarse_probabilities = ibs.wbia_plugin_curvrank_v2_coarse_probabilities(cropped_images)
        >>> coarse_probability = coarse_probabilities[0]
        >>> assert ut.hash_data(coarse_probability) in ['qnusxayrvvygnvllicwgeroesouxdfkh']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> aid_list *= 10
        >>> _, cropped_images, _ = ibs.wbia_plugin_curvrank_v2_preprocessing(aid_list)
        >>> coarse_probabilities = ibs.wbia_plugin_curvrank_v2_coarse_probabilities(cropped_images)
        >>> coarse_probability = coarse_probabilities[0]
        >>> assert ut.hash_data(coarse_probability) in ['qnusxayrvvygnvllicwgeroesouxdfkh']
    """
    model_tag = 'coarse.%s' % (model_type, )

    if model_tag in MODEL_URL_DICT:
        archive_url = MODEL_URL_DICT[model_tag]
        coarse_params = ut.grab_file_url(
            archive_url, appname='curvrank_v2', check_hash=True
        )
    else:
        raise RuntimeError

    unet = fcnn.UNet()
    device = get_device()
    unet.load_state_dict(torch.load(coarse_params, map_location=device))
    if torch.cuda.is_available():
        unet.cuda(None)
    unet.eval()
    coarse_probabilities = []
    for index, x in enumerate(cropped_images):
        x = cv2.resize(x, (width_coarse, height_coarse), interpolation=cv2.INTER_AREA)
        x = x.transpose(2, 0, 1) / 255.0
        x = x[np.newaxis, ...]
        x = torch.FloatTensor(x)
        if torch.cuda.is_available():
            x = x.cuda(None)
        with torch.no_grad():
            _, y_hat = unet(x)
        y_hat = y_hat.data.cpu().numpy().transpose(0, 2, 3, 1)
        probs = (255 * y_hat[0, :, :, 1]).astype(np.uint8)
        coarse_probabilities.append(probs)
    return coarse_probabilities


@register_ibs_method
def wbia_plugin_curvrank_v2_fine_probabilities(
    ibs,
    images,
    cropped_images,
    cropped_bboxes,
    coarse_probabilities,
    width_coarse=384,
    height_coarse=192,
    width_fine=1152,
    height_fine=576,
    patch_size=128,
    model_type='fluke',
    **kwargs):
    """
    Extract fine probabilities for CurvRank

    Args: #TODO
        ibs             (IBEISController): IBEIS controller object
        images          (list of np.ndarray): BGR images
        cropped_images  (list of np.ndarray): BGR images
        cropped_bboxes  (list of tuples)
        coarse_probabilities
        width_coarse    (int)
        height_coarse   (int)
        width_fine      (int)
        height_fine     (int)
        patch_size      (int)

    Returns:
        fine_probabilities

    CommandLine:
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_fine_probabilities
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_fine_probabilities:0
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_fine_probabilities:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> images, cropped_images, cropped_bboxes = ibs.wbia_plugin_curvrank_v2_preprocessing(aid_list)
        >>> coarse_probabilities = ibs.wbia_plugin_curvrank_v2_coarse_probabilities(cropped_images)
        >>> fine_probabilities = ibs.wbia_plugin_curvrank_v2_fine_probabilities(images, cropped_images, cropped_bboxes, coarse_probabilities)
        >>> fine_probability = fine_probabilities[0]
        >>> assert ut.hash_data(fine_probability) in ['vnlujxwbtwejjmvmsqwitopeoqejchdm']


    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> aid_list *= 10
        >>> images, cropped_images, cropped_bboxes = ibs.wbia_plugin_curvrank_v2_preprocessing(aid_list)
        >>> coarse_probabilities = ibs.wbia_plugin_curvrank_v2_coarse_probabilities(cropped_images)
        >>> fine_probabilities = ibs.wbia_plugin_curvrank_v2_fine_probabilities(images, cropped_images, cropped_bboxes, coarse_probabilities)
        >>> fine_probability = fine_probabilities[0]
        >>> assert ut.hash_data(fine_probability) in ['vnlujxwbtwejjmvmsqwitopeoqejchdm']
    """
    config_ = {
        'ordered': True,
        'chunksize': CHUNKSIZE,
        'force_serial': False,
        'progkw': {'freq': 10},
    }

    if model_type == 'fluke':
        generator = ut.generate2(
            F.control_points,
            zip(coarse_probabilities),
            nTasks=len(coarse_probabilities),
            **config_
        )
        control_points = []
        for cp in generator:
            control_points.append(cp)
        
        model_tag = 'fine.%s' % (model_type, )

        if model_tag in MODEL_URL_DICT:
            archive_url = MODEL_URL_DICT[model_tag]
            patch_params = ut.grab_file_url(
                archive_url, appname='curvrank_v2', check_hash=True
            )
        else:
            raise RuntimeError

        device = get_device()
        fine_probs = F.refine_by_network(
                        images,
                        cropped_images,
                        cropped_bboxes,
                        control_points,
                        width_coarse,
                        height_coarse,
                        width_fine,
                        height_fine,
                        patch_size,
                        patch_params,
                        device)

    elif model_type == 'dorsal':
        generator = ut.generate2(F.refine_by_gradient, zip(cropped_images), nTasks=len(cropped_images), **config_)
        fine_probs = []
        for fine_prob in generator:
            fine_probs.append(fine_prob)
    else:
        raise RuntimeError

    return fine_probs


@register_ibs_method
def wbia_plugin_curvrank_v2_anchor_points(
    ibs, cropped_images, width_fine=1152, width_anchor=224, height_anchor=224, model_type='fluke', **kwargs
):
    r"""
    Extract anchor points for CurvRank

    Args:
        ibs             (IBEISController): IBEIS controller object
        cropped_images  (list of np.ndarray): BGR images
        width_fine      (int): width of resized fine probabilities
        width_anchor    (int): width of network input
        height_anchor   (int): height of network input

    Returns:
        anchor_points

    CommandLine:
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_anchor_points
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_anchor_points:0
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_anchor_points:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> _, cropped_images, _ = ibs.wbia_plugin_curvrank_v2_preprocessing(aid_list)
        >>> coarse_probabilities = ibs.wbia_plugin_curvrank_v2_coarse_probabilities(cropped_images)
        >>> anchor_points = ibs.wbia_plugin_curvrank_v2_anchor_points(cropped_images)
        >>> start = np.around(anchor_points[0]['start'], 2).tolist()
        >>> end = np.around(anchor_points[0]['end'], 2).tolist()
        >>> assert start == [[24.73, 44.67]]
        >>> assert end == [[1073.62, 24.86]]

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> aid_list *= 10
        >>> _, cropped_images, _ = ibs.wbia_plugin_curvrank_v2_preprocessing(aid_list)
        >>> coarse_probabilities = ibs.wbia_plugin_curvrank_v2_coarse_probabilities(cropped_images)
        >>> anchor_points = ibs.wbia_plugin_curvrank_v2_anchor_points(cropped_images)
        >>> start = np.around(anchor_points[0]['start'], 2).tolist()
        >>> end = np.around(anchor_points[0]['end'], 2).tolist()
        >>> assert start == [[24.73, 44.67]]
        >>> assert end == [[1073.62, 24.86]]
    """
    model_tag = 'anchor.%s' % (model_type, )

    if model_tag in MODEL_URL_DICT:
        archive_url = MODEL_URL_DICT[model_tag]
        anchor_params = ut.grab_file_url(
            archive_url, appname='curvrank_v2', check_hash=True
        )
    else:
        raise RuntimeError

    anchor_nn = regression.VGG16()
    device = get_device()
    anchor_nn.load_state_dict(torch.load(anchor_params, map_location=device))
    if torch.cuda.is_available():
        anchor_nn.cuda(None)
    anchor_nn.eval()
    anchor_points = []
    for index, x in enumerate(cropped_images):
        part_img = x

        x = cv2.resize(x, (width_anchor, height_anchor), interpolation=cv2.INTER_AREA)
        x = x[:, :, ::-1] / 255.0
        x -= np.array([0.485, 0.456, 0.406])
        x /= np.array([0.229, 0.224, 0.225])
        x = x.transpose(2, 0, 1)
        x = x[np.newaxis, ...]
        x = torch.FloatTensor(x)
        if torch.cuda.is_available():
            x = x.cuda(None)
        with torch.no_grad():
            y0_hat, y1_hat = anchor_nn(x)
        y0_hat = y0_hat.data.cpu().numpy()
        y1_hat = y1_hat.data.cpu().numpy()

        ratio = width_fine / part_img.shape[1]
        part_img_resized = cv2.resize(
            part_img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA
        )
        height, width = part_img_resized.shape[0:2]
        start = y0_hat * np.array([width, height])
        end = y1_hat * np.array([width, height])
        anchor_points.append({'start': start, 'end': end})

    return anchor_points


@register_ibs_method
def wbia_plugin_curvrank_v2_contours(
    ibs,
    cropped_images,
    coarse_probabilities,
    fine_probabilities,
    anchor_points,
    trim=0,
    width_fine=1152,
    cost_func='exp',
    **kwargs
):
    r"""
    Extract contours for CurvRank

    Args:
        ibs                   (IBEISController): IBEIS controller object
        cropped_images        (list of np.ndarray): BGR images
        coarse_probabilities  (list of np.ndarray): Grayscale images
        fine_probabilities    (list of np.ndarray): Grayscale images
        anchor_points         (list of dicts): contour start and end points
        trim                  (int): number of points to trim from contour ends
        width_fine            (int): width of resized fine gradients
        csot_func             (str): type of cost function to use

    Returns:
        contours

    CommandLine:
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_contours
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_contours:0
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_contours:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> images, cropped_images, cropped_bboxes = ibs.wbia_plugin_curvrank_v2_preprocessing(aid_list)
        >>> coarse_probabilities = ibs.wbia_plugin_curvrank_v2_coarse_probabilities(cropped_images)
        >>> fine_probabilities = ibs.wbia_plugin_curvrank_v2_fine_probabilities(images, cropped_images, cropped_bboxes, coarse_probabilities)
        >>> anchor_points = ibs.wbia_plugin_curvrank_v2_anchor_points(cropped_images)
        >>> contours = ibs.wbia_plugin_curvrank_v2_contours(cropped_images, coarse_probabilities, fine_probabilities, anchor_points)
        >>> contour = contours[0]
        >>> assert ut.hash_data(contour) in ['jluhaoxjgacguizqrxyjvpglbscshlfv']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> aid_list *= 10
        >>> images, cropped_images, cropped_bboxes = ibs.wbia_plugin_curvrank_v2_preprocessing(aid_list)
        >>> coarse_probabilities = ibs.wbia_plugin_curvrank_v2_coarse_probabilities(cropped_images)
        >>> fine_probabilities = ibs.wbia_plugin_curvrank_v2_fine_probabilities(images, cropped_images, cropped_bboxes, coarse_probabilities)
        >>> anchor_points = ibs.wbia_plugin_curvrank_v2_anchor_points(cropped_images)
        >>> contours = ibs.wbia_plugin_curvrank_v2_contours(cropped_images, coarse_probabilities, fine_probabilities, anchor_points)
        >>> contour = contours[0]
        >>> assert ut.hash_data(contour) in ['jluhaoxjgacguizqrxyjvpglbscshlfv']
    """
    trim_list = [trim] * len(cropped_images)
    width_fine_list = [width_fine] * len(cropped_images)
    cost_func_list = [cost_func] * len(cropped_images)
    zipped = zip(
        cropped_images,
        coarse_probabilities,
        fine_probabilities,
        anchor_points,
        trim_list,
        width_fine_list,
        cost_func_list,
    )

    config_ = {
        'ordered': True,
        'chunksize': CHUNKSIZE,
        'force_serial': False,
        'progkw': {'freq': 10},
    }
    generator = ut.generate2(
        F.contour_from_anchorpoints, zipped, nTasks=len(cropped_images), **config_
    )

    contours = []
    for contour in generator:
        contours.append(contour)

    return contours


@register_ibs_method
def wbia_plugin_curvrank_v2_curvatures(
    ibs,
    contours,
    width_fine=1152,
    height_fine=576,
    scales=DEFAULT_SCALES['fluke'],
    transpose_dims=True,
    **kwargs
):
    r"""
    Extract curvatures for CurvRank

    Args:
        ibs                   (IBEISController): IBEIS controller object
        contours: output of wbia_plugin_curvrank_v2_contours
        width_fine            (int): width of resized fine gradients
        height_fine           (int): height of resized fine gradients
        scales                (list of floats): integral curvature scales
        transpose_dims        (bool): if True move contour start point from left to top

    Returns:
        curvatures

    CommandLine:
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_curvatures
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_curvatures:0
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_curvatures:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> images, cropped_images, cropped_bboxes = ibs.wbia_plugin_curvrank_v2_preprocessing(aid_list)
        >>> coarse_probabilities = ibs.wbia_plugin_curvrank_v2_coarse_probabilities(cropped_images)
        >>> fine_probabilities = ibs.wbia_plugin_curvrank_v2_fine_probabilities(images, cropped_images, cropped_bboxes, coarse_probabilities)
        >>> anchor_points = ibs.wbia_plugin_curvrank_v2_anchor_points(cropped_images)
        >>> contours = ibs.wbia_plugin_curvrank_v2_contours(cropped_images, coarse_probabilities, fine_probabilities, anchor_points)
        >>> curvatures = ibs.wbia_plugin_curvrank_v2_curvatures(contours)
        >>> curvature = curvatures[0]
        >>> assert ut.hash_data(curvature) in ['byjahrxbzgfkoatkpikcsejvoltxzqid']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> aid_list *= 10
        >>> images, cropped_images, cropped_bboxes = ibs.wbia_plugin_curvrank_v2_preprocessing(aid_list)
        >>> coarse_probabilities = ibs.wbia_plugin_curvrank_v2_coarse_probabilities(cropped_images)
        >>> fine_probabilities = ibs.wbia_plugin_curvrank_v2_fine_probabilities(images, cropped_images, cropped_bboxes, coarse_probabilities)
        >>> anchor_points = ibs.wbia_plugin_curvrank_v2_anchor_points(cropped_images)
        >>> contours = ibs.wbia_plugin_curvrank_v2_contours(cropped_images, coarse_probabilities, fine_probabilities, anchor_points)
        >>> curvatures = ibs.wbia_plugin_curvrank_v2_curvatures(contours)
        >>> curvature = curvatures[0]
        >>> assert ut.hash_data(curvature) in ['byjahrxbzgfkoatkpikcsejvoltxzqid']
    """
    height_fine_list = [height_fine] * len(contours)
    width_fine_list = [width_fine] * len(contours)
    scales_list = [scales] * len(contours)
    transpose_dims_list = [transpose_dims] * len(contours)

    zipped = zip(
        contours, width_fine_list, height_fine_list, scales_list, transpose_dims_list
    )

    config_ = {
        'ordered': True,
        'chunksize': CHUNKSIZE,
        'force_serial': False,
        'progkw': {'freq': 10},
    }
    generator = ut.generate2(F.curvature, zipped, nTasks=len(contours), **config_)

    curvatures = []
    for curvature in generator:
        curvatures.append(curvature)

    return curvatures


@register_ibs_method
def wbia_plugin_curvrank_v2_descriptors(
    ibs,
    contours,
    curvatures,
    scales=DEFAULT_SCALES['fluke'],
    curv_length=1024,
    feat_dim=32,
    num_keypoints=32,
    **kwargs
):
    r"""
    Extract descriptors for CurvRank

    Args:
        ibs            (IBEISController): IBEIS controller object
        contours: output of wbia_plugin_curvrank_v2_contours
        curvatures: output of wbia_plugin_curvrank_v2_curvatures
        scales         (list of floats): integral curvature scales
        curv_length    (int)
        feat_dim       (int): Descriptor dimentions
        num_keypoints  (int)

    Returns:
        curvatures

    CommandLine:
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_descriptors
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_descriptors:0
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_descriptors:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> images, cropped_images, cropped_bboxes = ibs.wbia_plugin_curvrank_v2_preprocessing(aid_list)
        >>> coarse_probabilities = ibs.wbia_plugin_curvrank_v2_coarse_probabilities(cropped_images)
        >>> fine_probabilities = ibs.wbia_plugin_curvrank_v2_fine_probabilities(images, cropped_images, cropped_bboxes, coarse_probabilities)
        >>> anchor_points = ibs.wbia_plugin_curvrank_v2_anchor_points(cropped_images)
        >>> contours = ibs.wbia_plugin_curvrank_v2_contours(cropped_images, coarse_probabilities, fine_probabilities, anchor_points)
        >>> curvatures = ibs.wbia_plugin_curvrank_v2_curvatures(contours)
        >>> values = ibs.wbia_plugin_curvrank_v2_descriptors(contours, curvatures)
        >>> success_list, descriptors = values
        >>> success = success_list[0]
        >>> curvature_descriptor_dict = descriptors[0]
        >>> hash_list = [
        >>>     ut.hash_data(curvature_descriptor_dict[scale])
        >>>     for scale in sorted(list(curvature_descriptor_dict.keys()))
        >>> ]
        >>> assert success == True
        >>> assert ut.hash_data(hash_list) in ['ghvpdcfvrvukasxpsoxhzjwyjbbxjzjv']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> aid_list *= 10
        >>> images, cropped_images, cropped_bboxes = ibs.wbia_plugin_curvrank_v2_preprocessing(aid_list)
        >>> coarse_probabilities = ibs.wbia_plugin_curvrank_v2_coarse_probabilities(cropped_images)
        >>> fine_probabilities = ibs.wbia_plugin_curvrank_v2_fine_probabilities(images, cropped_images, cropped_bboxes, coarse_probabilities)
        >>> anchor_points = ibs.wbia_plugin_curvrank_v2_anchor_points(cropped_images)
        >>> contours = ibs.wbia_plugin_curvrank_v2_contours(cropped_images, coarse_probabilities, fine_probabilities, anchor_points)
        >>> curvatures = ibs.wbia_plugin_curvrank_v2_curvatures(contours)
        >>> values = ibs.wbia_plugin_curvrank_v2_descriptors(contours, curvatures)
        >>> success_list, descriptors = values
        >>> success = success_list[0]
        >>> curvature_descriptor_dict = descriptors[0]
        >>> hash_list = [
        >>>     ut.hash_data(curvature_descriptor_dict[scale])
        >>>     for scale in sorted(list(curvature_descriptor_dict.keys()))
        >>> ]
        >>> assert success == True
        >>> assert ut.hash_data(hash_list) in ['ghvpdcfvrvukasxpsoxhzjwyjbbxjzjv']
    """
    scales_list = [scales] * len(contours)
    curv_length_list = [curv_length] * len(contours)
    feat_dim_list = [feat_dim] * len(contours)
    num_keypoints_list = [num_keypoints] * len(contours)

    zipped = zip(
        contours,
        curvatures,
        scales_list,
        curv_length_list,
        feat_dim_list,
        num_keypoints_list,
    )

    config_ = {
        'ordered': True,
        'chunksize': CHUNKSIZE,
        'force_serial': False,
        'progkw': {'freq': 10},
    }
    generator = ut.generate2(F.curvature_descriptors, zipped, **config_)

    descriptors, success_list = [], []
    for success, descriptor in generator:
        descriptors.append(descriptor)
        success_list.append(success)

    return success_list, descriptors


@register_ibs_method
def wbia_plugin_curvrank_v2_pipeline_compute(ibs, aid_list, config={}):
    r"""
    Args:
        ibs       (IBEISController): IBEIS controller object
        aid_list  (list of int): list of annotation rowids (aids)

    Returns:
        success_list
        descriptors

    CommandLine:
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_pipeline_compute
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_pipeline_compute:0
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_pipeline_compute:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> values = ibs.wbia_plugin_curvrank_v2_pipeline_compute(aid_list)
        >>> success_list, curvature_descriptor_dicts = values
        >>> curvature_descriptor_dict = curvature_descriptor_dicts[0]
        >>> assert success_list == [True]
        >>> hash_list = [
        >>>     ut.hash_data(curvature_descriptor_dict[scale])
        >>>     for scale in sorted(list(curvature_descriptor_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['ghvpdcfvrvukasxpsoxhzjwyjbbxjzjv']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> aid_list *= 20
        >>> values = ibs.wbia_plugin_curvrank_v2_pipeline_compute(aid_list)
        >>> success_list, curvature_descriptor_dicts = values
        >>> success_list = success_list[:1]
        >>> curvature_descriptor_dicts = curvature_descriptor_dicts[:1]
        >>> curvature_descriptor_dict = curvature_descriptor_dicts[0]
        >>> assert success_list == [True]
        >>> hash_list = [
        >>>     ut.hash_data(curvature_descriptor_dict[scale])
        >>>     for scale in sorted(list(curvature_descriptor_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['ghvpdcfvrvukasxpsoxhzjwyjbbxjzjv']
    """
    images, cropped_images, cropped_bboxes = ibs.wbia_plugin_curvrank_v2_preprocessing(
        aid_list, **config
    )

    coarse_probabilities = ibs.wbia_plugin_curvrank_v2_coarse_probabilities(
        cropped_images, **config
    )

    endpoints = ibs.wbia_plugin_curvrank_v2_anchor_points(cropped_images, **config)

    fine_probabilities = ibs.wbia_plugin_curvrank_v2_fine_probabilities(
        images, cropped_images, cropped_bboxes, coarse_probabilities, **config
    )

    contours = ibs.wbia_plugin_curvrank_v2_contours(
        cropped_images, coarse_probabilities, fine_probabilities, endpoints, **config
    )

    curvatures = ibs.wbia_plugin_curvrank_v2_curvatures(contours, **config)

    values = ibs.wbia_plugin_curvrank_v2_descriptors(contours, curvatures, **config)
    success_list, descriptors = values

    return success_list, descriptors


@register_ibs_method
def wbia_plugin_curvrank_v2_pipeline_aggregate(
    ibs, aid_list, success_list, descriptor_dict_list
):
    r"""
    Args:
        ibs       (IBEISController): IBEIS controller object
        aid_list  (list of int): list of annotation rowids (aids)
        success_list: output of wbia_plugin_curvrank_v2_compute
        descriptor_dict_list: output of wbia_plugin_curvrank_v2_compute

    Returns:
        lnbnn_dict

    CommandLine:
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_pipeline_aggregate
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_pipeline_aggregate:0

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> values = ibs.wbia_plugin_curvrank_v2_pipeline_compute(aid_list)
        >>> success_list, curvature_descriptor_dicts = values
        >>> lnbnn_dict = ibs.wbia_plugin_curvrank_v2_pipeline_aggregate(aid_list, success_list, curvature_descriptor_dicts)
        >>> hash_list = [
        >>>     ut.hash_data(lnbnn_dict[scale])
        >>>     for scale in sorted(list(lnbnn_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['httuhfgjmiobakvfjxoeifyemhpmukyl']
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
def wbia_plugin_curvrank_v2_pipeline(
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
        ibs                 (IBEISController): IBEIS controller object
        imageset_rowid      (int)
        aid_list            (list of ints)
        config              (dict)
        use_depc            (bool)
        use_depc_optimized  (bool)
        verbose             (bool)

    Returns:
        lnbnn_dict
        aid_list

    CommandLine:
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_pipeline
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_pipeline:0
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_pipeline:1
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_pipeline:2

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> lnbnn_dict, aid_list = ibs.wbia_plugin_curvrank_v2_pipeline(aid_list=aid_list, use_depc=False, use_depc_optimized=False)
        >>> hash_list = [
        >>>     ut.hash_data(lnbnn_dict[scale])
        >>>     for scale in sorted(list(lnbnn_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['httuhfgjmiobakvfjxoeifyemhpmukyl']

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> lnbnn_dict, aid_list = ibs.wbia_plugin_curvrank_v2_pipeline(aid_list=aid_list, use_depc=True, use_depc_optimized=False)
        >>> hash_list = [
        >>>     ut.hash_data(lnbnn_dict[scale])
        >>>     for scale in sorted(list(lnbnn_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['httuhfgjmiobakvfjxoeifyemhpmukyl']

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(23)
        >>> lnbnn_dict, aid_list = ibs.wbia_plugin_curvrank_v2_pipeline(aid_list=aid_list, use_depc=True, use_depc_optimized=True)
        >>> hash_list = [
        >>>     ut.hash_data(lnbnn_dict[scale])
        >>>     for scale in sorted(list(lnbnn_dict.keys()))
        >>> ]
        >>> assert ut.hash_data(hash_list) in ['httuhfgjmiobakvfjxoeifyemhpmukyl']
    """
    if aid_list is None:
        aid_list = ibs.get_imageset_aids(imageset_rowid)

    # Compute Curvature Descriptors
    if verbose:
        print('\tCompute Curvature V2 Pipeline')
    if use_depc:
        config_ = _convert_kwargs_config_to_depc_config(config)
        table_name = (
            'curvature_descriptor_optimized_two' if use_depc_optimized else 'curvature_descriptor_two'
        )
        success_list = ibs.depc_annot.get(table_name, aid_list, 'success', config=config_)
        descriptor_dict_list = ibs.depc_annot.get(
            table_name, aid_list, 'descriptor', config=config_
        )
    else:
        values = ibs.wbia_plugin_curvrank_v2_pipeline_compute(aid_list, config=config)
        success_list, descriptor_dict_list = values

    if verbose:
        print('\tAggregate Pipeline Results')

    lnbnn_dict = ibs.wbia_plugin_curvrank_v2_pipeline_aggregate(
        aid_list, success_list, descriptor_dict_list
    )

    return lnbnn_dict, aid_list


@register_ibs_method
def wbia_plugin_curvrank_v2_scores(
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
    Compute CurvRank scores

    Args:
        ibs                 (IBEISController): IBEIS controller object
        db_aid_list         (list of ints): database annotation rowids
        qr_aids_list        (list of ints): query annotation rowids
        config              (dict)
        verbose             (bool)
        use_names           (bool)
        minimum_score       (float)
        use_depc            (bool)
        use_depc_optimized  (bool)

    Returns:
        score_dict

    CommandLine:
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_scores
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_scores:0
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_scores:1
        python -m wbia_curvrank_v2._plugin --test-wbia_plugin_curvrank_v2_scores:2

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> db_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Fluke Database')
        >>> db_aid_list = ibs.get_imageset_aids(db_imageset_rowid)
        >>> qr_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Fluke Query')
        >>> qr_aid_list = ibs.get_imageset_aids(qr_imageset_rowid)
        >>> score_dict_iter = ibs.wbia_plugin_curvrank_v2_scores(db_aid_list, [qr_aid_list], use_depc=False)
        >>> score_dict_list = list(score_dict_iter)
        >>> qr_aid_list, score_dict = score_dict_list[0]
        >>> for key in score_dict:
        >>>     score_dict[key] = round(score_dict[key], 2)
        >>> result = sorted(score_dict.items())
        >>> print(result)
        [(7, -0.5), (8, -0.36), (9, -0.22), (10, -0.32), (11, -6.4), (12, -0.53), (13, -0.92), (14, -0.68)]

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> db_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Fluke Database')
        >>> db_aid_list = ibs.get_imageset_aids(db_imageset_rowid)
        >>> qr_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Fluke Query')
        >>> qr_aid_list = ibs.get_imageset_aids(qr_imageset_rowid)
        >>> score_dict_iter = ibs.wbia_plugin_curvrank_v2_scores(db_aid_list, [qr_aid_list], use_depc=True, use_depc_optimized=False)
        >>> score_dict_list = list(score_dict_iter)
        >>> qr_aid_list, score_dict = score_dict_list[0]
        >>> for key in score_dict:
        >>>     score_dict[key] = round(score_dict[key], 2)
        >>> result = sorted(score_dict.items())
        >>> print(result)
        [(7, -0.5), (8, -0.36), (9, -0.22), (10, -0.32), (11, -6.4), (12, -0.53), (13, -0.92), (14, -0.68)]

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from wbia_curvrank_v2._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_curvrank()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> db_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Fluke Database')
        >>> db_aid_list = ibs.get_imageset_aids(db_imageset_rowid)
        >>> qr_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Fluke Query')
        >>> qr_aid_list = ibs.get_imageset_aids(qr_imageset_rowid)
        >>> score_dict_iter = ibs.wbia_plugin_curvrank_v2_scores(db_aid_list, [qr_aid_list], use_depc=True, use_depc_optimized=True)
        >>> score_dict_list = list(score_dict_iter)
        >>> qr_aid_list, score_dict = score_dict_list[0]
        >>> for key in score_dict:
        >>>     score_dict[key] = round(score_dict[key], 2)
        >>> result = sorted(score_dict.items())
        >>> print(result)
        [(7, -0.5), (8, -0.36), (9, -0.22), (10, -0.32), (11, -6.4), (12, -0.53), (13, -0.92), (14, -0.68)]
    """
    cache_path = abspath(join(ibs.get_cachedir(), 'curvrank_v2'))
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
        'CurvRank V2 cache config:\n\tuse_daily_cache = %r\n\tdaily_cache_tag = %r\n\tforce_cache_recompute = %r\n\t'
        % args
    )
    print('CurvRank V2 num_trees   : %r' % (num_trees,))
    print('CurvRank V2 search_k    : %r' % (search_k,))
    print('CurvRank V2 lnbnn_k     : %r' % (lnbnn_k,))
    print('CurvRank V2 algo config : %s' % (ut.repr3(config),))

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
                print(
                    'Checking %r (%r)'
                    % (
                        directory,
                        then,
                    )
                )

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
                print(
                    'Checking %r (%r)'
                    % (
                        directory,
                        then,
                    )
                )

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
                % (
                    force_cache_recompute,
                    index_directory,
                )
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
                % (
                    num_trees_,
                    num_trees,
                    num_annots,
                )
            )
            print(
                '[global] WARNING! Using search_k = %d instead of %d (based on %d annotations)'
                % (
                    search_k_,
                    search_k,
                    num_annots,
                )
            )
            num_trees = num_trees_
            search_k = search_k_

    index_path = join(cache_path, index_directory)

    with ut.Timer('Loading query'):
        scale_set = set([])
        qr_lnbnn_data_list = []
        for qr_aid_list in ut.ProgressIter(
            qr_aids_list, lbl='CurvRank V2 Query LNBNN', freq=1000
        ):
            values = ibs.wbia_plugin_curvrank_v2_pipeline(
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
                                % (
                                    num_trees_,
                                    num_trees,
                                    num_annots,
                                )
                            )
                            print(
                                '[local] WARNING! Using search_k = %d instead of %d (based on %d annotations)'
                                % (
                                    search_k_,
                                    search_k,
                                    num_annots,
                                )
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
            future_index_directory = '%s%s' % (
                FUTURE_PREFIX,
                index_directory,
            )
            future_index_path = join(cache_path, future_index_directory)
            ut.ensuredir(future_index_path)

            with ut.Timer('Loading database LNBNN descriptors from depc'):
                values = ibs.wbia_plugin_curvrank_v2_pipeline(
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
                            % (
                                scale,
                                future_index_filepath,
                            )
                        )
                        descriptors, aids = db_lnbnn_data[scale]
                        F.build_lnbnn_index(
                            descriptors, future_index_filepath, num_trees=num_trees
                        )
                    else:
                        ut.copy(index_filepath, future_index_filepath)
                        print(
                            'Using existing Annoy scale=%r index in %r...'
                            % (
                                scale,
                                index_filepath,
                            )
                        )

                    if not exists(aids_filepath):
                        print(
                            'Writing computed AIDs scale=%r to %r...'
                            % (
                                scale,
                                future_aids_filepath,
                            )
                        )
                        ut.save_cPkl(future_aids_filepath, aids)
                        print('\t...saved')
                    else:
                        ut.copy(aids_filepath, future_aids_filepath)
                        print(
                            'Using existing AIDs scale=%r in %r...'
                            % (
                                scale,
                                aids_filepath,
                            )
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
            zipped, lbl='CurvRank V2 Vectored Scoring', freq=1000
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
def wbia_plugin_curvrank_v2(ibs, label, qaid_list, daid_list, config):
    r"""
    Compute CurvRank scores

    Args:
        ibs        (IBEISController): IBEIS controller object
        label      (string)
        qaid_list  (list of ints): query annotaion rowids
        daid_list  (list of ints): database annotaion rowids
        config     (dict)

    CommandLine:
        python -m wbia_curvrank_v2._plugin --exec-wbia_plugin_curvrank_v2

    Example:
        >>> # ENABLE_DOCTEST
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
        >>> # Call function normally
        >>> config = CurvRankFlukeConfig()
        >>> score_list = list(ibs.wbia_plugin_curvrank_v2('CurvRankTest', qaid_list, daid_list, config))
        >>> score_list = [round(score[0], 5) for score in score_list]
        >>> result = score_list
        >>> print(result)
        [-0.0, 0.39218, 0.10824, 0.05452, 0.04381, 0.02845, 0.01364, 0.02443, 0.006, 0.01736, 0.02502, 0.00436, 0.03939, 0.02246, 0.01645, 0.0268, 0.01983, 0.06692, 0.04013, 0.1084, 0.1497, 0.04716, 0.05244, 0.2237, 0.1728, 0.00796, 0.0123, 0.09124, 0.09705, 0.06993, 0.01437, 0.68972, -0.0, 0.06153, 0.00468, 0.01922, 0.03674, 0.02334, 0.00968, 0.03023, 0.02984, 0.04094, 0.02096, 0.04407, 0.04911, 0.0181, 0.01312, 0.01601, 0.05483, 0.06387, 0.20325, 0.46781, 0.06959, 0.12506, 0.33767, 0.38326, 0.00655, 0.00297, 0.06257, 0.19678, 0.04586, 0.03737, 0.8752, 0.13795, -0.0, 0.11811, 0.10333, 0.04353, 0.04367, 0.07554, 0.07422, 0.06055, 0.10653, 0.02263, 0.05414, 0.02677, 0.09502, 0.04009, 0.13963, 0.03198, 0.06115, 0.$8712, 0.05517, 0.16054, 0.21613, 0.09592, 0.1589, 0.06624, 0.07657, 0.28956, 0.03136, 0.10778, 0.04377, 0.19586, 0.08591, 0.07865, -0.0, 0.54172, 0.39211, 0.06883, 0.05905, 0.07558, 0.02464, 0.0768, 0.04013, 0.12865, 0.03713, 0.12362, 0.$4299, 0.10507, 0.07551, 0.11166, 0.1021, 0.09371, 0.19675, 0.06888, 0.14543, 0.16085, 0.04259, 0.03809, 0.04827, 0.08515, 0.12129, 0.03583, 0.21865, 0.08862, 0.04791, 0.9184, -0.0, 3.12676, 0.03012, 0.02869, 0.018, 0.01657, 0.05353, 0.03$52, 0.05029, 0.02024, 0.02783, 0.03178, 0.03417, 0.01825, 0.02977, 0.11659, 0.09621, 0.04746, 0.05178, 0.07715, 0.06541, 0.01865, 0.01844, 0.19661, 0.04877, 0.05344, 0.01091, 0.14554, 0.05673, 0.03192, 0.40825, 2.93053, -0.0, 0.15526, 0.$375, 0.07169, 0.01688, 0.04947, 0.03123, 0.02007, 0.04976, 0.09654, 0.03144, 0.08321, 0.04364, 0.10701, 0.12525, 0.05065, 0.10945, 0.10359, 0.15628, 0.12164, 0.01615, 0.04693, 0.05602, 0.09072, 0.06254, 0.01248, 0.06113, 0.07714, 0.03251$ 0.03257, 0.02622, 0.37584, -0.0, 1.51071, 1.77567, 0.28317, 0.09435, 0.2727, 0.04009, 0.01972, 0.04304, 0.08269, 0.02237, 0.05965, 0.06139, 0.06087, 0.07459, 0.09268, 0.08251, 0.12794, 0.05679, 0.02087, 0.0405, 0.07013, 0.0299, 0.03682, 0.01223, 0.05274, 0.09789, 0.04883, 0.09772, 0.05682, 0.05609, 1.44907, -0.0, 2.2163, 0.02689, 0.06509, 0.05282, 0.06843, 0.02514, 0.06739, 0.03656, 0.05772, 0.03369, 0.06668, 0.06163, 0.07695, 0.05461, 0.03266, 0.07809, 0.06434, 0.01289$ 0.01602, 0.09038, 0.04785, 0.04328, 0.03653, 0.04376, 0.05944, 0.02236, 0.03133, 0.01268, 0.11344, 1.57602, 2.78007, -0.0, 0.02896, 0.03388, 0.01628, 0.06065, 0.04719, 0.06421, 0.07635, 0.03877, 0.04784, 0.02007, 0.08904, 0.08044, 0.110$3, 0.04998, 0.10547, 0.06804, 0.02771, 0.02428, 0.07872, 0.04609, 0.03883, 0.05324, 0.02243, 0.02774, 0.01114, 0.029, 0.05785, 0.01551, 0.03045, 0.01202, 0.04019, -0.0, 5.13383, 3.06859, 0.04014, 0.03532, 0.04471, 0.04128, 0.02294, 0.036$5, 0.02294, 0.02787, 0.0428, 0.03056, 0.02795, 0.02109, 0.02145, 0.00525, 0.01723, 0.0305, 0.03237, 0.03084, 0.01834, 0.06268, 0.03643, 0.06408, 0.0603, 0.06348, 0.01484, 0.01247, 0.02376, 0.01904, 1.79269, -0.0, 4.06144, 0.0312, 0.01062$ 0.04808, 0.02229, 0.03451, 0.02337, 0.0202, 0.06482, 0.04968, 0.07644, 0.03908, 0.03818, 0.06261, 0.00301, 0.01696, 0.07174, 0.0381, 0.02105, 0.0467, 0.02506, 0.0276, 0.00856, 0.01921, 0.0358, 0.12983, 0.02606, 0.0096, 0.0094, 1.89537, $.32629, -0.0, 0.02748, 0.01275, 0.02013, 0.00475, 0.01741, 0.01519, 0.03414, 0.04576, 0.05532, 0.03338, 0.02246, 0.02166, 0.03761, 0.00874, 0.01289, 0.00973, 0.02377, 0.02751, 0.01654, 0.08032, 0.09308, 0.01517, 0.02135, 0.02514, 0.01468$ 0.01541, 0.00793, 0.01104, 0.01262, 0.02692, 0.02289, -0.0, 0.52534, 0.05793, 0.05562, 0.5663, 0.76496, 0.05923, 0.10132, 0.20057, 0.01444, 0.04007, 0.1963, 0.10541, 0.01017, 0.01242, 0.02254, 0.03595, 0.01252, 0.04378, 0.04136, 0.02778$ 0.00559, 0.0207, 0.00876, 0.02102, 0.00889, 0.00499, 0.00098, 0.03291, 0.01708, 0.00396, 0.08313, -0.0, 0.16169, 0.2585, 2.28728, 7.82047, 0.03639, 0.01402, 0.02088, 0.01455, 0.02874, 0.02757, 0.02396, 0.00771, 0.00447, 0.01733, 0.00803$ 0.01619, 0.02575, 0.08982, 0.07146, 0.02342, 0.05082, 0.01836, 0.04084, 0.05406, 0.01689, 0.02241, 0.03474, 0.05941, 0.02829, 0.24455, 0.72881, -0.0, 0.26335, 0.19823, 0.96966, 0.07053, 0.04561, 0.04536, 0.08274, 0.06373, 0.12653, 0.072$8, 0.00525, 0.02381, 0.03199, 0.04195, 0.07092, 0.01415, 0.03715, 0.02502, 0.03764, 0.01441, 0.03001, 0.05501, 0.00849, 0.00746, 0.06477, 0.01908, 0.01066, 0.00973, 0.30408, 3.11227, 0.29877, -0.0, 0.77604, 1.24962, 0.05031, 0.02359, 0.0$752, 0.03334, 0.03229, 0.02205, 0.02846, 0.00295, 0.00763, 0.00743, 0.0122, 0.01695, 0.03921, 0.02731, 0.0149, 0.01106, 0.04375, 0.01306, 0.0031, 0.00687, 0.01362, 0.00282, 0.01889, 0.01037, 0.03538, 0.55228, 3.36366, 0.08521, 0.34238, -$.0, 3.79058, 0.03652, 0.03491, 0.06253, 0.00495, 0.04324, 0.05046, 0.04323, 0.02301, 0.00624, 0.00673, 0.00697, 0.01663, 0.02171, 0.0087, 0.02368, 0.01266, 0.01468, 0.01141, 0.02904, 0.00367, 0.01023, 0.01146, 0.01107, 0.00515, 0.01505, $.31068, 8.30201, 0.22366, 0.32658, 2.39195, -0.0, 0.00806, 0.0265, 0.02114, 0.00499, 0.00469, 0.04828, 0.02606, 0.01226, 0.00285, 0.00742, 0.01166, 0.04178, 0.10513, 0.02316, 0.04599, 0.015, 0.02265, 0.01043, 0.0232, 0.01835, 0.02443, 0.$0944, 0.01646, 0.01992, 0.00532, 0.0706, 0.01172, 0.03823, 0.00987, 0.02167, 0.02197, -0.0, 4.05023, 3.48772, 0.0153, 0.02585, 0.11058, 0.04452, 0.02002, 0.0157, 0.00145, 0.02459, 0.03806, 0.01052, 0.06091, 0.05572, 0.00795, 0.02102, 0.0$948, 0.01193, 0.00554, 0.00339, 0.01037, 0.01725, 0.01132, 0.00527, 0.02907, 0.00575, 0.01391, 0.01957, 0.03273, 0.01966, 2.82016, -0.0, 1.13898, 0.03222, 0.05935, 0.17775, 0.11474, 0.00335, 0.00404, 0.01281, 0.02493, 0.03244, 0.00075, 0$08562, 0.19483, 0.01761, 0.01078, 0.0417, 0.02277, 0.01519, 0.02207, 0.00543, 0.0211, 0.0137, 0.00973, 0.02908, 0.0132, 0.00572, 0.02138, 0.01178, 0.01853, 0.85468, 0.47263, -0.0, 0.00527, 0.02288, 0.14891, 0.28802, 0.01034, 0.00356, 0.0$738, 0.12646, 0.04304, 0.0027, 0.07512, 0.04189, 0.04098, 0.00708, 0.00966, 0.02401, 0.02852, 0.02587, 0.01938, 0.0062, 0.02188, 0.01581, 0.02916, 0.0273, 0.07528, 0.02073, 0.01981, 0.01467, 0.01961, 0.0409, 0.04097, -0.0, 4.77879, 0.672$4, 0.05201, 0.00771, 0.01378, 0.01478, 0.01031, 0.02323, 0.01747, 0.07852, 0.07606, 0.01915, 0.02554, 0.02851, 0.06247, 0.00658, 0.01518, 0.00746, 0.0082, 0.0218, 0.00803, 0.03988, 0.03105, 0.06562, 0.01426, 0.04292, 0.02032, 0.02489, 0.$0789, 0.06344, 4.20461, -0.0, 1.06766, 0.06342, 0.00608, 0.0116, 0.02957, 0.03984, 0.00668, 0.00435, 0.06018, 0.14487, 0.01597, 0.00483, 0.01293, 0.01233, 0.01232, 0.00824, 0.01346, 0.00233, 0.00948, 0.00759, 0.06574, 0.01415, 0.02574, 0$00386, 0.03186, 0.03101, 0.06792, 0.19753, 0.20561, 0.06894, 0.54445, -0.0, 0.11088, 0.0024, 0.00153, 0.01118, 0.02584, 0.02742, 0.04259, 0.07914, 0.13925, 0.03697, 0.02459, 0.02113, 0.03163, 0.02293, 0.01197, 0.01018, 0.00405, 0.02044, $.02066, 0.03862, 0.01556, 0.00715, 0.03242, 0.02579, 0.04705, 0.0356, 0.13632, 0.1995, 0.02462, 0.01921, 0.14445, -0.0, 0.00617, 0.00777, 0.19631, 0.97621, 0.1603, 0.01831, 0.08004, 0.06294, 0.12643, 0.11982, 0.06445, 0.08437, 0.05614, 0$10558, 0.02995, 0.0701, 0.04194, 0.02406, 0.03861, 0.07655, 0.16322, 0.0802, 0.02758, 0.05226, 0.04656, 0.01487, 0.02221, 0.02987, 0.07837, 0.07189, 0.21615, -0.0, 0.54746, 2.61184, 0.24486, 0.16259, 0.25932, 0.08827, 0.08004, 0.10127, 0$09588, 0.08661, 0.23705, 0.04477, 0.06225, 0.04805, 0.08372, 0.0568, 0.03397, 0.08005, 0.04097, 0.10461, 0.05567, 0.05226, 0.04219, 0.07015, 0.05853, 0.01634, 0.03699, 0.09742, 0.07555, 0.09082, 0.13788, -0.0, 0.23465, 0.86254, 1.88719, $.105, 0.16535, 0.03519, 0.06096, 0.03438, 0.09691, 0.02554, 0.04159, 0.03293, 0.01503, 0.0121, 0.01759, 0.01264, 0.04142, 0.03668, 0.04997, 0.05392, 0.0246, 0.02912, 0.05371, 0.07012, 0.08768, 0.02868, 0.06577, 0.06853, 2.66722, 0.11473,0.01867, -0.0, 0.49973, 0.1333, 0.02771, 0.04388, 0.04783, 0.00331, 4e-05, 0.00748, 0.00733, 0.01068, 0.00653, 0.01613, 0.00666, 0.02176, 0.01613, 0.01824, 0.01328, 0.00361, 0.01587, 0.00298, 0.03286, 0.03643, 0.04697, 0.18127, 0.00972, $.00477, 0.06784, 1.55499, 0.00455, 0.03677, 0.11688, -0.0, 3.32981, 0.02387, 0.0641, 0.13722, 0.03339, 0.0239, 0.02388, 0.04075, 0.00998, 0.01056, 0.01848, 0.01504, 0.02062, 0.02623, 0.01225, 0.01368, 0.04291, 0.02736, 0.0306, 0.02976, 0$035, 0.03363, 0.07677, 0.02789, 0.01839, 0.07675, 0.42813, 0.00129, 0.02157, 0.0918, 6.28942, -0.0, 0.09264, 0.04528, 0.08742, 0.10998, 0.04513, 0.07754, 0.0957, 0.07876, 0.09463, 0.04283, 0.048, 0.06098, 0.03675, 0.06156, 0.0509, 0.0976$, 0.06618, 0.07924, 0.08969, 0.08579, 0.0962, 0.09831, 0.05245, 0.12778, 0.07325, 0.08379, 0.15438, 0.0719, 0.2549, 0.14044, 0.25151, -0.0]
    """
    print('Computing %s' % (label,))

    cache_path = abspath(join(ibs.get_cachedir(), 'curvrank_v2'))
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
        'Computing IBEIS CurvRank V2 (%s) on %d total qaids (%d unique), %d total daids (%d unique)'
        % args
    )
    with ut.Timer(message):
        value_iter = ibs.wbia_plugin_curvrank_v2_scores_depc(
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
        zipped, 'CurvRank V2 Pair-wise Final Scores', freq=1000
    ):
        assert qaid in score_dict
        score = score_dict[qaid].get(daid, 0.0)
        score *= -1.0

        yield (score,)


@register_ibs_method
def wbia_plugin_curvrank_v2_delete_cache_optimized(ibs, aid_list, tablename):
    import networkx as nx

    assert tablename in [
        'CurvRankTwoDorsal',
        'CurvRankTwoFluke',
    ]

    tablename_list = [
        'curvature_descriptor_optimized_two',
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
        python -m wbia_curvrank_v2._plugin --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
