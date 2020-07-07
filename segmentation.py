# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from lasagne.layers import batch_norm
from lasagne.layers import Conv2DLayer
from lasagne.layers import Deconv2DLayer
from lasagne.layers import InputLayer
from lasagne.layers import get_all_layers
from lasagne.init import Orthogonal
from lasagne.nonlinearities import rectify, sigmoid


def build_model_batchnorm_full(input_shape):
    common = {'W': Orthogonal(gain='relu'), 'nonlinearity': rectify}

    num_filters = 32
    l_in = InputLayer(input_shape, name='seg_in')

    # 256 x 256
    l_conv1 = batch_norm(Conv2DLayer(
        l_in, name='seg_conv1', num_filters=num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_conv2 = batch_norm(Conv2DLayer(
        l_conv1, name='seg_conv2', num_filters=num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_conv3 = batch_norm(Conv2DLayer(
        l_conv2, name='seg_conv3', num_filters=2 * num_filters,
        filter_size=(2, 2), stride=(2, 2), **common
    ))

    # 128 x 128
    l_conv4 = batch_norm(Conv2DLayer(
        l_conv3, name='seg_conv4', num_filters=2 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_conv5 = batch_norm(Conv2DLayer(
        l_conv4, name='seg_conv5', num_filters=2 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_conv6 = batch_norm(Conv2DLayer(
        l_conv5, name='seg_conv6', num_filters=4 * num_filters,
        filter_size=(2, 2), stride=(2, 2), **common
    ))

    # 64 x 64
    l_conv7 = batch_norm(Conv2DLayer(
        l_conv6, name='seg_conv7', num_filters=4 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_conv8 = batch_norm(Conv2DLayer(
        l_conv7, name='seg_conv8', num_filters=4 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_conv9 = batch_norm(Conv2DLayer(
        l_conv8, name='seg_conv9', num_filters=8 * num_filters,
        filter_size=(2, 2), stride=(2, 2), **common
    ))

    # 32 x 32
    l_conv10 = batch_norm(Conv2DLayer(
        l_conv9, name='seg_conv10', num_filters=8 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_conv11 = batch_norm(Conv2DLayer(
        l_conv10, name='seg_conv11', num_filters=8 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_conv12 = batch_norm(Deconv2DLayer(
        l_conv11, name='seg_conv12', num_filters=4 * num_filters,
        filter_size=(2, 2), stride=(2, 2), **common
    ))

    # 64 x 64
    l_conv13 = batch_norm(Conv2DLayer(
        l_conv12, name='seg_conv13', num_filters=4 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_conv14 = batch_norm(Conv2DLayer(
        l_conv13, name='seg_conv14', num_filters=4 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_conv15 = batch_norm(Deconv2DLayer(
        l_conv14, name='seg_conv15', num_filters=2 * num_filters,
        filter_size=(2, 2), stride=(2, 2), **common
    ))

    # 128 x 128
    l_conv16 = batch_norm(Conv2DLayer(
        l_conv15, name='seg_conv16', num_filters=2 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_conv17 = batch_norm(Conv2DLayer(
        l_conv16, name='seg_conv17', num_filters=2 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_conv18 = batch_norm(Deconv2DLayer(
        l_conv17, name='seg_conv18', num_filters=num_filters,
        filter_size=(2, 2), stride=(2, 2), **common
    ))

    # 256 x 256
    l_conv19 = batch_norm(Conv2DLayer(
        l_conv18, name='seg_conv19', num_filters=num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_out = Conv2DLayer(
        l_conv19, name='seg_out', num_filters=1,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2,
        nonlinearity=sigmoid, W=Orthogonal(gain=1.0),
    )

    model_layers = get_all_layers(l_out)
    return {layer.name: layer for layer in model_layers}


def build_model_batchnorm(input_shape):
    common = {'W': Orthogonal(gain='relu'), 'nonlinearity': rectify}

    num_filters = 32
    l_in = InputLayer(input_shape, name='seg_in')

    # 128 x 128
    l_conv1 = batch_norm(Conv2DLayer(
        l_in, name='seg_conv1', num_filters=num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_conv2 = batch_norm(Conv2DLayer(
        l_conv1, name='seg_conv2', num_filters=num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_conv3 = batch_norm(Conv2DLayer(
        l_conv2, name='seg_conv3', num_filters=2 * num_filters,
        filter_size=(2, 2), stride=(2, 2), **common
    ))

    # 64 x 64
    l_conv4 = batch_norm(Conv2DLayer(
        l_conv3, name='seg_conv4', num_filters=2 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_conv5 = batch_norm(Conv2DLayer(
        l_conv4, name='seg_conv5', num_filters=2 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_conv6 = batch_norm(Conv2DLayer(
        l_conv5, name='seg_conv6', num_filters=4 * num_filters,
        filter_size=(2, 2), stride=(2, 2), **common
    ))

    # 32 x 32
    l_conv7 = batch_norm(Conv2DLayer(
        l_conv6, name='seg_conv7', num_filters=4 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_conv8 = batch_norm(Conv2DLayer(
        l_conv7, name='seg_conv8', num_filters=4 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_conv9 = batch_norm(Deconv2DLayer(
        l_conv8, name='seg_conv9', num_filters=2 * num_filters,
        filter_size=(2, 2), stride=(2, 2), **common
    ))

    # 64 x 64
    l_conv10 = batch_norm(Conv2DLayer(
        l_conv9, name='seg_conv10', num_filters=2 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_conv11 = batch_norm(Conv2DLayer(
        l_conv10, name='seg_conv11', num_filters=2 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_conv12 = batch_norm(Deconv2DLayer(
        l_conv11, name='seg_conv12', num_filters=num_filters,
        filter_size=(2, 2), stride=(2, 2), **common
    ))

    # 128 x 128
    l_conv13 = batch_norm(Conv2DLayer(
        l_conv12, name='seg_conv13', num_filters=num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2, **common
    ))
    l_out = Conv2DLayer(
        l_conv13, name='seg_out', num_filters=1,
        filter_size=(3, 3), stride=(1, 1), pad=3 // 2,
        nonlinearity=sigmoid, W=Orthogonal(gain=1.0),
    )

    model_layers = get_all_layers(l_out)
    return {layer.name: layer for layer in model_layers}


def test_build_model_batchnorm():
    from wbia_curvrank import model
    layer_dict = build_model_batchnorm((None, 3, 128, 128))
    model.print_info(layer_dict['seg_out'])


def test_full_size_inference():
    from wbia_curvrank import model, theano_funcs
    import cv2
    import numpy as np
    from os.path import join
    layer_dict = build_model_batchnorm((None, 3, None, None))
    segmentation_weightsfile = join(
        'weights', 'weights_segmentation_all.pickle'
    )
    print('loading weights for the segmentation network from %s' % (
        segmentation_weightsfile))
    model.load_weights(
        layer_dict['seg_out'],
        segmentation_weightsfile
    )
    infer_func = theano_funcs.create_segmentation_infer_func(layer_dict)
    img_hwc = cv2.imread('img.png')
    #img_hwc = cv2.resize(img_hwc, (128, 128))
    print('img.shape = %r' % (img_hwc.shape,))
    img_chw = (img_hwc / 255.).transpose(2, 0, 1).astype(np.float32)
    img_bchw = img_chw.reshape(
        -1, img_chw.shape[0], img_chw.shape[1], img_chw.shape[2])
    seg_bchw = infer_func(img_bchw)
    seg_hwc = seg_bchw[0].transpose(1, 2, 0)
    print('seg.shape = %r' % (seg_hwc.shape,))
    print('seg.min() = %.2f, seg.max() = %.2f' % (
        seg_hwc.min(), seg_hwc.max()))
    cv2.imwrite('seg.png', 255 * seg_hwc)


if __name__ == '__main__':
    #test_build_model_batchnorm()
    #test_full_size_inference()
    from wbia_curvrank import theano_funcs
    import numpy as np
    input_shape = (32, 3, 480, 128)
    X = np.random.random(input_shape).astype(np.float32)
    layer_dict = build_model_batchnorm_full(input_shape)
    infer_func = theano_funcs.create_segmentation_infer_func(layer_dict)
    Z = infer_func(X)
    print(Z.shape)
