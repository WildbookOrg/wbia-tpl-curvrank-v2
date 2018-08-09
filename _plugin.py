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


class CropChipConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('crop_dim_size', 750, 'sz', hideif=750),
            ut.ParamInfo('crop_enabled', True, hideif=False),
            #ut.ParamInfo('ccversion', 1)
            ut.ParamInfo('version', 2),
            ut.ParamInfo('ext', '.png'),
        ]


# Custom chip table
@register_preproc(
    'Cropped_Chips',
    parents=[const.CHIP_TABLE, 'Notch_Tips'],
    colnames=['img', 'width', 'height', 'M', 'notch', 'left', 'right'],
    coltypes=[('extern', vt.imread, vt.imwrite), int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    configclass=CropChipConfig,
    fname='cropped_chip'
)
def preproc_cropped_chips(depc, cid_list, tipid_list, config=None):
    pass
