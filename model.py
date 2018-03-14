from __future__ import absolute_import, division, print_function
import numpy as np
from lasagne.layers import get_all_params
from lasagne.layers import count_params
from lasagne.layers import get_all_layers
import six
if six.PY2:
    import cPickle as pickle
else:
    import pickle


def save_weights(weights, filename):
    with open(filename, 'wb') as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_weights(layer, filename):
    with open(filename, 'rb') as f:
        try:
            src_params_list = pickle.load(f)
        except UnicodeDecodeError:
            src_params_list = pickle.load(f, encoding='latin1')

    dst_params_list = get_all_params(layer)
    # assign the parameter values stored on disk to the model
    for src_params, dst_params in zip(src_params_list, dst_params_list):
        dst_params.set_value(src_params)


def print_info(layer):
    layers = get_all_layers(layer)
    print('this network has %d learnable parameters' % (
        (count_params(layer))))
    for layer in layers:
        num_params = 0
        if hasattr(layer, 'W') and layer.W is not None:
            num_params += np.prod(layer.W.get_value().shape)
        if hasattr(layer, 'b') and layer.b is not None:
            num_params += np.prod(layer.b.get_value().shape)
        print('%s: %s, %r, %d' % (
            layer.name, layer.__class__.__name__,
            layer.output_shape, num_params))
        #layer.name, layer, layer.output_shape, num_params))
