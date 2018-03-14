from __future__ import absolute_import, division, print_function
import numpy as np
from lasagne.layers import batch_norm
from lasagne.layers import Conv2DLayer
from lasagne.layers import DenseLayer
#from lasagne.layers import DropoutLayer
from lasagne.layers import InputLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers import TransformerLayer
from lasagne.layers import get_all_layers

from lasagne.nonlinearities import rectify, linear
from lasagne.init import Constant
from lasagne.init import Orthogonal


def build_model(input_shape, downsample=1):
    conv = {
        'filter_size': (3, 3), 'stride': (1, 1), 'pad': 1,
        'W': Orthogonal(gain='relu'), 'nonlinearity': rectify,
    }
    pool = {'mode': 'max', 'pool_size': (2, 2), 'stride': (2, 2)}
    dense = {
        'num_units': 256, 'W': Orthogonal(gain='relu'),
        'nonlinearity': rectify
    }

    # input layer for downsampled and padded input
    l_in = InputLayer(input_shape, name='in')

    # group 1
    l_conv1 = Conv2DLayer(
        l_in, name='loc_conv1', num_filters=32, **conv
    )
    l_pool1 = Pool2DLayer(l_conv1, name='loc_pool1', **pool)

    # group 2
    l_conv2a = Conv2DLayer(
        l_pool1, name='loc_conv2a', num_filters=64, **conv
    )
    l_conv2b = Conv2DLayer(
        l_conv2a, name='loc_conv2b', num_filters=64, **conv
    )
    l_pool2 = Pool2DLayer(l_conv2b, name='loc_pool2', **pool)

    # group 3
    l_conv3a = Conv2DLayer(
        l_pool2, name='loc_conv3a', num_filters=64, **conv
    )
    l_conv3b = Conv2DLayer(
        l_conv3a, name='loc_conv3b', num_filters=64, **conv
    )
    l_pool3 = Pool2DLayer(l_conv3b, name='loc_pool3', **pool)

    # group 4
    l_conv4a = Conv2DLayer(
        l_pool3, name='loc_conv4a', num_filters=128, **conv
    )
    l_conv4b = Conv2DLayer(
        l_conv4a, name='loc_conv4b', num_filters=128, **conv
    )
    l_pool4 = Pool2DLayer(l_conv4b, name='loc_pool4', **pool)

    # group 5
    l_conv5a = Conv2DLayer(
        l_pool4, name='loc_conv5a', num_filters=256, **conv
    )
    l_conv5b = Conv2DLayer(
        l_conv5a, name='loc_conv5b', num_filters=256, **conv
    )
    l_pool5 = Pool2DLayer(l_conv5b, name='loc_pool5', **pool)

    #l_drop5 = DropoutLayer(l_pool5, name='loc_drop5', p=0.5)
    l_dense5 = DenseLayer(
        l_pool5, name='loc_dense5', **dense
    )

    #l_drop6 = DropoutLayer(l_dense5, name='loc_dense5', p=0.5)
    l_dense6 = DenseLayer(
        l_dense5, name='loc_dense6', **dense
    )

    # important to initialize to the identity transform
    W = Constant(0.0)
    b = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32).flatten()
    l_loc = DenseLayer(
        l_dense6, num_units=6, W=W, b=b, name='loc', nonlinearity=linear
    )

    # transforms the downsampled input
    l_trans = TransformerLayer(
        l_in, l_loc, downsample_factor=downsample, name='trans'
    )

    model_layers = get_all_layers([l_loc, l_trans])
    return {layer.name: layer for layer in model_layers}


def build_model_batchnorm(input_shape, downsample=1):
    conv = {
        'filter_size': (3, 3), 'stride': (1, 1), 'pad': 1,
        'W': Orthogonal(gain='relu'), 'nonlinearity': rectify,
    }
    pool = {'mode': 'max', 'pool_size': (2, 2), 'stride': (2, 2)}
    dense = {
        'num_units': 256, 'W': Orthogonal(gain='relu'),
        'nonlinearity': rectify
    }

    l_in = InputLayer(input_shape, name='in')

    # group 1
    l_conv1 = batch_norm(Conv2DLayer(
        l_in, name='loc_conv1', num_filters=32, **conv
    ))
    l_pool1 = Pool2DLayer(l_conv1, name='loc_pool1', **pool)

    # group 2
    l_conv2a = batch_norm(Conv2DLayer(
        l_pool1, name='loc_conv2a', num_filters=64, **conv
    ))
    l_conv2b = batch_norm(Conv2DLayer(
        l_conv2a, name='loc_conv2b', num_filters=64, **conv
    ))
    l_pool2 = Pool2DLayer(l_conv2b, name='loc_pool2', **pool)

    # group 3
    l_conv3a = batch_norm(Conv2DLayer(
        l_pool2, name='loc_conv3a', num_filters=64, **conv
    ))
    l_conv3b = batch_norm(Conv2DLayer(
        l_conv3a, name='loc_conv3b', num_filters=64, **conv
    ))
    l_pool3 = Pool2DLayer(l_conv3b, name='loc_pool3', **pool)

    # group 4
    l_conv4a = batch_norm(Conv2DLayer(
        l_pool3, name='loc_conv4a', num_filters=128, **conv
    ))
    l_conv4b = batch_norm(Conv2DLayer(
        l_conv4a, name='loc_conv4b', num_filters=128, **conv
    ))
    l_pool4 = Pool2DLayer(l_conv4b, name='loc_pool4', **pool)

    # group 5
    l_conv5a = batch_norm(Conv2DLayer(
        l_pool4, name='loc_conv5a', num_filters=256, **conv
    ))
    l_conv5b = batch_norm(Conv2DLayer(
        l_conv5a, name='loc_conv5b', num_filters=256, **conv
    ))
    l_pool5 = Pool2DLayer(l_conv5b, name='loc_pool5', **pool)

    #l_drop5 = DropoutLayer(l_pool5, name='loc_drop5', p=0.5)
    l_dense5 = batch_norm(DenseLayer(
        l_pool5, name='loc_dense5', **dense
    ))

    #l_drop6 = DropoutLayer(l_dense5, name='loc_dense5', p=0.5)
    l_dense6 = batch_norm(DenseLayer(
        l_dense5, name='loc_dense6', **dense
    ))

    # important to initialize to the identity transform
    W = Constant(0.0)
    b = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32).flatten()
    l_loc = DenseLayer(
        l_dense6, num_units=6, W=W, b=b, name='loc', nonlinearity=linear
    )

    l_trans_down = TransformerLayer(
        l_in, l_loc, downsample_factor=downsample, name='trans'
    )

    model_layers = get_all_layers([l_loc, l_trans_down])
    return {layer.name: layer for layer in model_layers}


def test_build_model():
    from ibeis_curvrank import model
    layer_dict = build_model((None, 3, 256, 256), downsample=2)
    model.print_info([layer_dict['loc'], layer_dict['trans']])


if __name__ == '__main__':
    test_build_model()
