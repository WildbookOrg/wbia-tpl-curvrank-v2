import numpy as np


def transform_points(M, X):
    X = np.dot(M, np.hstack((X, np.ones((X.shape[0], 1)))).T)
    X = X[:2, :].T

    return X


def build_upsample_matrix(height, width):
    U = np.array([
        [(width - 1.) / 2., 0,                 (width - 1.) / 2.],
        [0.,               (height - 1.) / 2., (height - 1.) / 2.],
        [0.,               0.,                 1.]
    ], dtype=np.float32)

    return U


def build_downsample_matrix(height, width):
    D = np.array([
        [2. / (width - 1.), 0,                  -1],
        [0.,                2. / (height - 1.), -1],
        [0.,                0.,                  1.]
    ], dtype=np.float32)

    return D


def build_scale_matrix(s):
    S = np.array([
        [s,  0., 0.],
        [0., s,  0.],
        [0., 0., 1.]
    ], dtype=np.float32)

    return S


def multiply_matrices(L):
    I = np.eye(3)
    for A in L:
        if A.shape[0] == 2:
            A = np.vstack((A, np.array([0, 0, 1])))
        I = np.dot(I, A)

    return I


# https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/special.py#L463
def _interpolate(im, x, y, out_height, out_width):
    # *_f are floats
    num_batch, height, width, channels = im.shape
    height_f, width_f = float(height), float(width)

    # clip coordinates to [-1, 1]
    x = np.clip(x, -1, 1)
    y = np.clip(y, -1, 1)

    # scale coordinates from [-1, 1] to [0, width/height - 1]
    x = (x + 1) / 2 * (width_f - 1)
    y = (y + 1) / 2 * (height_f - 1)

    x0_f = np.floor(x)
    y0_f = np.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1
    x0 = x0_f.astype(np.int32)
    y0 = y0_f.astype(np.int32)
    x1 = np.minimum(x1_f, width_f - 1).astype(np.int32)
    y1 = np.minimum(y1_f, height_f - 1).astype(np.int32)

    # The input is [num_batch, height, width, channels]. We do the lookup in
    # the flattened input, i.e [num_batch*height*width, channels]. We need
    # to offset all indices to match the flat version
    dim2 = width
    dim1 = width * height
    base = np.repeat(
        np.arange(num_batch, dtype='int64') * dim1, out_height * out_width)
    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels for all samples
    im_flat = im.reshape((-1, channels))
    Ia = im_flat[idx_a]
    Ib = im_flat[idx_b]
    Ic = im_flat[idx_c]
    Id = im_flat[idx_d]

    # calculate interpolated values
    wa = ((x1_f - x) * (y1_f - y)).reshape(-1, 1)
    wb = ((x1_f - x) * (y - y0_f)).reshape(-1, 1)
    wc = ((x - x0_f) * (y1_f - y)).reshape(-1, 1)
    wd = ((x - x0_f) * (y - y0_f)).reshape(-1, 1)
    output = np.sum([wa * Ia, wb * Ib, wc * Ic, wd * Id], axis=0)
    return output


# https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/special.py#L528
def _meshgrid(height, width):
    # This function is the grid generator from eq. (1) in reference [1].
    # It is equivalent to the following numpy code:
    x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
                           np.linspace(-1, 1, height))
    ones = np.ones(np.prod(x_t.shape))
    grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    return grid


# https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/special.py#L463
def _transform_affine(theta, input, height, width):
    #num_batch, num_channels, height, width = input.shape
    input = input.transpose(2, 0, 1).reshape(-1, input.shape[2], input.shape[0], input.shape[1])
    num_batch, num_channels, _, _ = input.shape
    theta = theta.reshape(-1, 2, 3)

    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    #out_height = int(height // downsample_factor[0])
    #out_width = int(width // downsample_factor[1])
    out_height = height
    out_width = width
    grid = _meshgrid(out_height, out_width)

    # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
    T_g = np.dot(theta, grid)
    x_s = T_g[:, 0]
    y_s = T_g[:, 1]
    x_s_flat = x_s.flatten()
    y_s_flat = y_s.flatten()

    # dimshuffle input to  (bs, height, width, channels)
    input_dim = input.transpose(0, 2, 3, 1)
    input_transformed = _interpolate(
        input_dim, x_s_flat, y_s_flat,
        out_height, out_width)

    output = input_transformed.reshape(
        num_batch, out_height, out_width, num_channels)
    #output = output.transpose(0, 3, 1, 2)  # dimshuffle to conv format
    return output[0].astype(np.float32)
