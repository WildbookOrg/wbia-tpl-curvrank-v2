from __future__ import absolute_import, division, print_function
from ibeis.control import controller_inject  # NOQA
import ibeis.constants as const
import numpy as np
import utool as ut
import vtool as vt
import dtool

_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)
register_api = controller_inject.get_ibeis_flask_api(__name__)
register_preproc = controller_inject.register_preprocs['annot']


# @register_ibs_method
# @register_api('/api/plugin/curvrank/helloworld/', methods=['GET'])
# def ibeis_plugin_curvrank_hello_world(ibs, *args, **kwargs):
#     return args, kwargs


@register_ibs_method
def ibeis_plugin_curvrank_example(ibs):
    from ibeis_curvrank.example_workflow import example
    example()


@register_ibs_method
def ibeis_plugin_curvrank_aids(ibs, aid_list):

    results_list = []

    return results_list


@register_ibs_method
def ibeis_plugin_curvrank(ibs, image_filepath_list, name_list, flip_list):
    results_list = []
    return results_list


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
        >>> values = ibs.ibeis_plugin_curvrank_preprocessing(gid_list)
        >>> resized_images, resized_masks, pre_transforms = values
        >>> resized_images = resized_images[0]
        >>> resized_masks  = resized_masks[0]
        >>> pre_transforms = pre_transforms[0]
        >>> assert ut.hash_data(resized_images) == 'ynbsgqgfutslspmatpenvcbtgedsyzoo'
        >>> assert ut.hash_data(resized_masks) == 'mnhartnytowmmhskblocubqmzhbofynr'
        >>> result = pre_transforms
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


# class CropChipConfig(dtool.Config):
#     def get_param_info_list(self):
#         return [
#             ut.ParamInfo('crop_dim_size', 750, 'sz', hideif=750),
#             ut.ParamInfo('crop_enabled', True, hideif=False),
#             #ut.ParamInfo('ccversion', 1)
#             ut.ParamInfo('version', 2),
#             ut.ParamInfo('ext', '.png'),
#         ]

# # Custom chip table
# @register_preproc(
#     'Cropped_Chips',
#     parents=[const.ANNOT_TABLE],
#     colnames=['img', 'width', 'height', 'M', 'notch', 'left', 'right'],
#     coltypes=[('extern', vt.imread, vt.imwrite), int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
#     configclass=CropChipConfig,
#     fname='cropped_chip'
# )
# def preproc_cropped_chips(depc, cid_list, tipid_list, config=None):
#     pass


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
