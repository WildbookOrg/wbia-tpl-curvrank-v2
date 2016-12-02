from lasagne.layers import batch_norm
from lasagne.layers import Conv2DLayer
from lasagne.layers import Deconv2DLayer
from lasagne.layers import InputLayer
from lasagne.layers import get_all_layers
from lasagne.init import Orthogonal
from lasagne.nonlinearities import rectify, sigmoid, linear


def build_model_batchnorm_full(input_shape):
    common = {'W': Orthogonal(gain='relu'), 'nonlinearity': rectify}

    num_filters = 32
    l_in = InputLayer(input_shape, name='key_in')

    # 256 x 256
    l_conv1 = batch_norm(Conv2DLayer(
        l_in, name='key_conv1', num_filters=num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_conv2 = batch_norm(Conv2DLayer(
        l_conv1, name='key_conv2', num_filters=num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_conv3 = batch_norm(Conv2DLayer(
        l_conv2, name='key_conv3', num_filters=2 * num_filters,
        filter_size=(2, 2), stride=(2, 2), **common
    ))

    # 128 x 128
    l_conv4 = batch_norm(Conv2DLayer(
        l_conv3, name='key_conv4', num_filters=2 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_conv5 = batch_norm(Conv2DLayer(
        l_conv4, name='key_conv5', num_filters=2 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_conv6 = batch_norm(Conv2DLayer(
        l_conv5, name='key_conv6', num_filters=4 * num_filters,
        filter_size=(2, 2), stride=(2, 2), **common
    ))

    # 64 x 64
    l_conv7 = batch_norm(Conv2DLayer(
        l_conv6, name='key_conv7', num_filters=4 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_conv8 = batch_norm(Conv2DLayer(
        l_conv7, name='key_conv8', num_filters=4 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_conv9 = batch_norm(Conv2DLayer(
        l_conv8, name='key_conv9', num_filters=8 * num_filters,
        filter_size=(2, 2), stride=(2, 2), **common
    ))

    # 32 x 32
    l_conv10 = batch_norm(Conv2DLayer(
        l_conv9, name='key_conv10', num_filters=8 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_conv11 = batch_norm(Conv2DLayer(
        l_conv10, name='key_conv11', num_filters=8 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_conv12 = batch_norm(Deconv2DLayer(
        l_conv11, name='key_conv12', num_filters=4 * num_filters,
        filter_size=(2, 2), stride=(2, 2), **common
    ))

    # 64 x 64
    l_conv13 = batch_norm(Conv2DLayer(
        l_conv12, name='key_conv13', num_filters=4 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_conv14 = batch_norm(Conv2DLayer(
        l_conv13, name='key_conv14', num_filters=4 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_conv15 = batch_norm(Deconv2DLayer(
        l_conv14, name='key_conv15', num_filters=2 * num_filters,
        filter_size=(2, 2), stride=(2, 2), **common
    ))

    # 128 x 128
    l_conv16 = batch_norm(Conv2DLayer(
        l_conv15, name='key_conv16', num_filters=2 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_conv17 = batch_norm(Conv2DLayer(
        l_conv16, name='key_conv17', num_filters=2 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_conv18 = batch_norm(Deconv2DLayer(
        l_conv17, name='key_conv18', num_filters=num_filters,
        filter_size=(2, 2), stride=(2, 2), **common
    ))

    # 256 x 256
    l_conv19 = batch_norm(Conv2DLayer(
        l_conv18, name='key_conv19', num_filters=num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_out = Conv2DLayer(
        l_conv19, name='key_out', num_filters=3,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2,
        nonlinearity=sigmoid, W=Orthogonal(gain=1.0),
    )

    model_layers = get_all_layers(l_out)
    return {layer.name: layer for layer in model_layers}


def build_model_batchnorm(input_shape):
    common = {'W': Orthogonal(gain='relu'), 'nonlinearity': rectify}

    num_filters = 32
    l_in = InputLayer(input_shape, name='key_in')

    # 128 x 128
    l_conv1 = batch_norm(Conv2DLayer(
        l_in, name='key_conv1', num_filters=num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_conv2 = batch_norm(Conv2DLayer(
        l_conv1, name='key_conv2', num_filters=num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_conv3 = batch_norm(Conv2DLayer(
        l_conv2, name='key_conv3', num_filters=2 * num_filters,
        filter_size=(2, 2), stride=(2, 2), **common
    ))

    # 64 x 64
    l_conv4 = batch_norm(Conv2DLayer(
        l_conv3, name='key_conv4', num_filters=2 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_conv5 = batch_norm(Conv2DLayer(
        l_conv4, name='key_conv5', num_filters=2 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_conv6 = batch_norm(Conv2DLayer(
        l_conv5, name='key_conv6', num_filters=4 * num_filters,
        filter_size=(2, 2), stride=(2, 2), **common
    ))

    # 32 x 32
    l_conv7 = batch_norm(Conv2DLayer(
        l_conv6, name='key_conv7', num_filters=4 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_conv8 = batch_norm(Conv2DLayer(
        l_conv7, name='key_conv8', num_filters=4 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_conv9 = batch_norm(Deconv2DLayer(
        l_conv8, name='key_conv9', num_filters=2 * num_filters,
        filter_size=(2, 2), stride=(2, 2), **common
    ))

    # 64 x 64
    l_conv10 = batch_norm(Conv2DLayer(
        l_conv9, name='key_conv10', num_filters=2 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_conv11 = batch_norm(Conv2DLayer(
        l_conv10, name='key_conv11', num_filters=2 * num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_conv12 = batch_norm(Deconv2DLayer(
        l_conv11, name='key_conv12', num_filters=num_filters,
        filter_size=(2, 2), stride=(2, 2), **common
    ))

    # 128 x 128
    l_conv13 = batch_norm(Conv2DLayer(
        l_conv12, name='key_conv13', num_filters=num_filters,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2, **common
    ))
    l_out = Conv2DLayer(
        l_conv13, name='key_out', num_filters=3,
        #l_conv13, name='key_out', num_filters=1,
        filter_size=(3, 3), stride=(1, 1), pad=3 / 2,
        #nonlinearity=linear, W=Orthogonal(gain=1.0),
        nonlinearity=sigmoid, W=Orthogonal(gain=1.0),
    )

    model_layers = get_all_layers(l_out)
    return {layer.name: layer for layer in model_layers}


def test_build_model_batchnorm():
    import model
    layers = build_model_batchnorm((None, 3, 128, 128))
    model.print_info(layers['key_out'])


def test_build_model_batchnorm_inverted():
    import model
    layers = build_model_batchnorm_inverted((None, 3, 128, 128))
    model.print_info(layers['key_out'])


if __name__ == '__main__':
    #test_build_model_batchnorm_inverted()
    test_build_model_batchnorm()
