from __future__ import absolute_import, division, print_function
import numpy as np
import theano
import theano.tensor as T

from lasagne.updates import nesterov_momentum
from lasagne.layers import get_all_params
from lasagne.layers import get_output


def create_localization_train_func(layers, lr=0.01, mntm=0.9):
    X = T.tensor4('X')
    X_batch = T.tensor4('X_batch')
    Y = T.tensor4('Y')
    Y_batch = T.tensor4('Y_batch')
    Z = T.btensor4('Z')
    Z_batch = T.btensor4('Z_batch')

    _, X_hat = get_output(
        [layers['loc'], layers['trans']],
        X, deterministic=False)

    # compute the loss between the transformed image and the given target
    #train_loss = T.mean(
    #    T.mean(T.sqr(X_hat - Y), axis=1)
    #)
    #train_loss = T.mean(T.sqr(X_hat - Y) * Z)
    train_loss = T.sum(T.sqr(X_hat - Y) * Z) / Z.sum()

    params = get_all_params([layers['loc'], layers['trans']], trainable=True)

    updates = nesterov_momentum(train_loss, params, lr, mntm)

    train_func = theano.function(
        inputs=[theano.In(X_batch), theano.In(Y_batch), theano.In(Z_batch)],
        outputs=[train_loss, X_hat],
        updates=updates,
        givens={
            X: X_batch,
            Y: Y_batch,
            Z: Z_batch,
        },
    )

    return train_func


def create_localization_valid_func(layers):
    X = T.tensor4('X')
    X_batch = T.tensor4('X_batch')
    Y = T.tensor4('Y')
    Y_batch = T.tensor4('Y_batch')
    Z = T.btensor4('Z')
    Z_batch = T.btensor4('Z_batch')

    _, X_hat = get_output(
        [layers['loc'], layers['trans']],
        X, deterministic=True)

    # compute the loss between the transformed image and the given target
    #valid_loss = T.mean(
    #    T.mean(T.sqr(X_hat - Y), axis=1)
    #)
    #valid_loss = T.mean(T.sqr(X_hat - Y) * Z)
    valid_loss = T.sum(T.sqr(X_hat - Y) * Z) / Z.sum()

    valid_func = theano.function(
        inputs=[theano.In(X_batch), theano.In(Y_batch), theano.In(Z_batch)],
        outputs=[valid_loss, X_hat],
        updates=None,
        givens={
            X: X_batch,
            Y: Y_batch,
            Z: Z_batch,
        },
    )

    return valid_func


def create_localization_infer_func(layers):
    X = T.tensor4('X')
    X_batch = T.tensor4('X_batch')

    M, X_hat = get_output(
        [layers['loc'], layers['trans']],
        X, deterministic=True)

    infer_func = theano.function(
        inputs=[theano.In(X_batch)],
        outputs=[M, X_hat],
        updates=None,
        givens={
            X: X_batch,
        },
    )

    return infer_func


def create_localization_test_func(layers):
    X_full = T.tensor4('X_full')
    X_full_batch = T.tensor4('X_full_batch')
    X_down = T.tensor4('X_down')
    X_down_batch = T.tensor4('X_down_batch')

    X_hat_down, X_hat_full = get_output(
        [layers['trans'], layers['trans_full']],
        inputs={layers['in']: X_down, layers['in_full']: X_full},
        deterministic=True
    )

    test_func = theano.function(
        inputs=[theano.In(X_down_batch), theano.In(X_full_batch)],
        outputs=[X_hat_down, X_hat_full],
        givens={
            X_down: X_down_batch,
            X_full: X_full_batch,
        }
    )

    return test_func


def create_segmentation_train_func(layers, lr=0.01, mntm=0.9):
    X = T.tensor4('X')
    X_batch = T.tensor4('X_batch')
    Y = T.tensor4('Y')
    Y_batch = T.tensor4('Y_batch')

    X_hat = get_output(layers['seg_out'], X, deterministic=False)

    #train_loss = T.mean(
    #    T.mean(T.sqr(X_hat - Y), axis=1)
    #)
    train_loss = T.mean(
        T.nnet.binary_crossentropy(
            T.clip(X_hat, 1e-15, 1 - 1e-15),
            Y)
    )

    params = get_all_params(layers['seg_out'], trainable=True)

    updates = nesterov_momentum(train_loss, params, lr, mntm)

    train_func = theano.function(
        inputs=[theano.In(X_batch), theano.In(Y_batch)],
        outputs=[train_loss, X_hat],
        updates=updates,
        givens={
            X: X_batch,
            Y: Y_batch,
        },
    )

    return train_func


def create_segmentation_valid_func(layers):
    X = T.tensor4('X')
    X_batch = T.tensor4('X_batch')
    Y = T.tensor4('Y')
    Y_batch = T.tensor4('Y_batch')

    X_hat = get_output(layers['seg_out'], X, deterministic=True)

    #valid_loss = T.mean(
    #    T.mean(T.sqr(X_hat - Y), axis=1)
    #)
    valid_loss = T.mean(
        T.nnet.binary_crossentropy(
            T.clip(X_hat, 1e-15, 1 - 1e-15),
            Y)
    )

    valid_func = theano.function(
        inputs=[theano.In(X_batch), theano.In(Y_batch)],
        outputs=[valid_loss, X_hat],
        updates=None,
        givens={
            X: X_batch,
            Y: Y_batch,
        },
    )

    return valid_func


def create_segmentation_infer_func(layers):
    X = T.tensor4('X')
    X_batch = T.tensor4('X_batch')

    X_hat = get_output(layers['seg_out'], X, deterministic=True)

    infer_func = theano.function(
        inputs=[theano.In(X_batch)],
        outputs=X_hat,
        updates=None,
        givens={
            X: X_batch,
        },
    )

    return infer_func


def create_segmentation_func(layers):
    X = T.tensor4('X')
    X_batch = T.tensor4('X_batch')

    # final segmentation
    S  = get_output(layers['seg_out'], X, deterministic=True)

    infer_func = theano.function(
        inputs=[theano.In(X_batch)],
        outputs=S,
        updates=None,
        givens={
            X: X_batch,
        },
    )

    return infer_func


def test_localization_funcs():
    from ibeis_curvrank import localization
    print('testing localization')
    print('  building model')
    layers = localization.build_model((None, 3, 256, 256), downsample=2)

    print('  compiling training function')
    loc_train_func = create_localization_train_func(layers)
    print('  compiling validation function')
    loc_valid_func = create_localization_valid_func(layers)
    print('  compiling inference function')
    loc_infer_func = create_localization_infer_func(layers)

    X = np.random.random((16, 3, 256, 256)).astype(np.float32)
    Y = np.random.random((16, 3, 128, 128)).astype(np.float32)
    Z = np.random.randint(0, 2, (16, 3, 128, 128)).astype(np.int32)

    print('  forward/backward pass')
    train_loss, _ = loc_train_func(X, Y, Z)
    print('    train loss = %.6f' % (train_loss))

    print('  forward pass with loss')
    valid_loss, _ = loc_valid_func(X, Y, Z)
    print('    valid loss = %.6f' % (valid_loss))

    print('  forward pass without loss')
    M, X_hat = loc_infer_func(X)
    print('M.shape = %r, X_hat.shape = %r' % (M.shape, X_hat.shape))

    print('done testing localization')


def test_segmentation_funcs():
    from ibeis_curvrank import segmentation
    print('testing segmentation')
    print('  building model')
    layers = segmentation.build_model((None, 3, 128, 128))
    print('  compiling training function')
    seg_train_func = create_segmentation_train_func(layers)
    print('  compiling validation function')
    seg_valid_func = create_segmentation_valid_func(layers)
    print('  compiling inference function')
    seg_infer_func = create_segmentation_infer_func(layers)

    X = np.random.random((16, 3, 128, 128)).astype(np.float32)
    Y = np.random.random((16, 3, 128, 128)).astype(np.float32)

    print('  forward/backward pass')
    train_loss, _ = seg_train_func(X, Y)
    print('    train loss = %.6f' % (train_loss))

    print('  forward pass with loss')
    valid_loss, _ = seg_valid_func(X, Y)
    print('    valid loss = %.6f' % (valid_loss))

    print('  forward pass without loss')
    X_hat = seg_infer_func(X)
    print('X_hat.shape = %r' % (X_hat.shape,))

    print('done testing segmentation')


if __name__ == '__main__':
    test_localization_funcs()
    test_segmentation_funcs()
