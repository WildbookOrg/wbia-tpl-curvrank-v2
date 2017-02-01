import cPickle as pickle
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import InputLayer
from lasagne.layers import ScaleLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import get_all_params
from lasagne.layers import get_all_param_values
from lasagne.layers import get_output
from lasagne.layers import get_all_layers
from lasagne.updates import nesterov_momentum
from lasagne.regularization import regularize_network_params, l2

import model
from dorsal_utils import resampleNd


def build_model(input_shape):
    batch_size, num_scales, num_points = input_shape
    l_in = InputLayer(input_shape, name='l_in')

    l_scale1 = ScaleLayer(
        l_in, scales=np.ones(num_points) / np.sqrt(num_points),
        shared_axes=(0, 1), name='spatial_weights',
    )
    l_norm1 = ExpressionLayer(
        l_scale1,
        lambda x: x / T.sqrt(T.sum(T.sqr(x), axis=2))[:, :, np.newaxis],
        name='l2norm'
    )
    l_scale2 = ScaleLayer(
        l_norm1, scales=np.ones(num_scales) / np.sqrt(num_scales),
        shared_axes=(0, 2), name='scale_weights'
    )

    layers = get_all_layers(l_scale2)
    return {layer.name: layer for layer in layers}


def encode_func(layers):
    lr, mntm = 0.1, 0.9

    X = T.tensor3('X')
    X_batch = T.tensor3('X_batch')
    y = T.ivector('y')
    y_batch = T.ivector('y_batch')

    F = get_output(layers['scale_weights'], X, deterministic=True)

    same = F[0::2]
    diff = F[1::2]

    dist = T.sum(T.sum(T.sqr(same - diff), axis=2), axis=1)
    margin = 20.
    dist_loss = T.mean(y * dist + (1 - y) * T.maximum(margin - dist, 0))

    l2_loss = regularize_network_params(layers['scale_weights'], l2)

    loss = dist_loss + 0.01 * l2_loss

    params = get_all_params(layers['scale_weights'], trainable=True)
    updates = nesterov_momentum(loss, params, lr, mntm)

    encode_func = theano.function(
        inputs=[theano.In(X_batch), theano.In(y_batch)],
        outputs=[F, loss, dist],
        updates=updates,
        givens={
            X: X_batch,
            y: y_batch,
        }
    )

    return encode_func


def sample(db_dict, curv_dict):
    indivs = db_dict.keys()

    X = np.empty((4 * len(indivs), 4, 128), dtype=np.float32)
    y = np.empty(2 * len(indivs), dtype=np.int32)

    for i, indiv in enumerate(indivs):
        refn_curv_fpaths = db_dict[indiv]
        refn_curv_fpath, same_curv_fpath =\
            np.random.choice(refn_curv_fpaths, 2)
        diff_curv_fpaths = db_dict[np.random.choice(
            [ind for ind in indivs if ind != indiv]
        )]
        diff_curv_fpath = np.random.choice(diff_curv_fpaths)

        with open(curv_dict[refn_curv_fpath]['curvature'].path, 'rb') as f1,\
                open(curv_dict[same_curv_fpath]['curvature'].path, 'rb') as f2,\
                open(curv_dict[diff_curv_fpath]['curvature'].path, 'rb') as f3:
            refn_curv = pickle.load(f1)
            same_curv = pickle.load(f2)
            diff_curv = pickle.load(f3)

        X[4 * i] = resampleNd(refn_curv, 128).T
        X[4 * i + 1] = resampleNd(same_curv, 128).T
        y[2 * i] = 1
        X[4 * i + 2] = resampleNd(refn_curv, 128).T
        X[4 * i + 3] = resampleNd(diff_curv, 128).T
        y[2 * i + 1] = 0

    return X, y


def main():
    from run_luigi import BlockCurvature
    curv_dict = BlockCurvature(
        dataset='sdrp', oriented=True,
        curvature_scales=(0.11, 0.16, 0.21, 0.26)
    ).output()

    db_fpath = 'data/sdrp/SeparateDatabaseQueries/database.pickle'
    with open(db_fpath, 'rb') as f:
        db_fpath_dict = pickle.load(f)

    num_scales, num_points = 4, 128
    input_shape = (None, num_scales, num_points)
    layers = build_model(input_shape)
    encode = encode_func(layers)
    #X = np.random.random((128, num_scales, num_points)).astype(np.float32)
    #y = np.random.randint(0, 2, 128 / 2).astype(np.int32)
    #X[np.repeat(y, 2)] += 2
    #print X
    try:
        for i in range(100000):
            X, y = sample(db_fpath_dict, curv_dict)
            encoding, loss, dist = encode(X, y)
            params = get_all_params(layers['scale_weights'])
            for p in params:
                p_val = p.get_value()
                #print p, np.linalg.norm(p.get_value())
                if np.linalg.norm(p_val) > 1.0:
                    p_val = p_val / np.sqrt(np.sum(p_val ** 2))
                    p.set_value(p_val)
                #print p, np.linalg.norm(p.get_value())
            print('loss = %.6f' % (loss))
            print('mean dist diff = %.6f, mean dist same = %.6f' % (
                dist[y == 0].mean(), dist[y == 1].mean()
            ))
    except KeyboardInterrupt:
        print('caught ctrl-c')

    print('weights for %d spatial points' % (num_points))
    print layers['spatial_weights'].scales.get_value()
    print('weights for %d scales' % (num_scales))
    print layers['scale_weights'].scales.get_value()

    weights = get_all_param_values(layers['scale_weights'])
    model.save_weights(weights, 'weights.pickle')

    #print encoding
    #print np.linalg.norm(encoding, axis=2)


if __name__ == '__main__':
    main()
