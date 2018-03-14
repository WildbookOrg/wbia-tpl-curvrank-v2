from __future__ import absolute_import, division, print_function
from ibeis.control import controller_inject  # NOQA


_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)
register_api = controller_inject.get_ibeis_flask_api(__name__)


@register_ibs_method
def ibeis_plugin_curvrank_example(ibs):
    from ibeis_curvrank.example_workflow import example
    example()


@register_api('/api/plugin/example/helloworld/', methods=['GET'])
def ibeis_plugin_example_hello_world_rest(ibs):
    return ibs.ibeis_plugin_example_hello_world()
