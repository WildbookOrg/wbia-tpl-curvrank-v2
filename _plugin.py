from __future__ import absolute_import, division, print_function
from ibeis.control import controller_inject  # NOQA
import ibeis.constants as const
import numpy as np
import utool as ut
import vtool as vt
import dtool

_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)
register_api = controller_inject.get_ibeis_flask_api(__name__)
register_preproc_image = controller_inject.register_preprocs['image']
register_preproc_annot = controller_inject.register_preprocs['annot']


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


class PreprocessConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('preprocess_height', 256),
            ut.ParamInfo('preprocess_width', 256),
            ut.ParamInfo('ext', '.png', hideif='.png'),
        ]


@register_preproc_image(
    tablename='preprocess', parents=['images'],
    colnames=['resized_img', 'resized_width', 'resized_height', 'mask_img', 'mask_width', 'mask_height', 'pretransform'],
    coltypes=[('extern', vt.imread, vt.imwrite), int, int, ('extern', ut.partial(vt.imread, grayscale=True), vt.imwrite), int, int, np.ndarray],
    configclass=PreprocessConfig,
    fname='preprocess',
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
        >>> assert ut.hash_data(resized_image) == 'ynbsgqgfutslspmatpenvcbtgedsyzoo'
        >>> assert ut.hash_data(resized_mask)  == 'mnhartnytowmmhskblocubqmzhbofynr'
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
        >>> assert ut.hash_data(resized_image) == 'ynbsgqgfutslspmatpenvcbtgedsyzoo'
        >>> assert ut.hash_data(resized_mask) == 'mnhartnytowmmhskblocubqmzhbofynr'
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
