import cv2
import functional as F
import localization
import model
import numpy as np
import segmentation
import theano_funcs
from dorsal_utils import find_dorsal_keypoints, dorsal_cost_func
from dorsal_utils import separate_leading_trailing_edges
from os.path import join


def example():
    fpath = '/media/sdrp/SDRP Data/FinBase/Images/11173.JPG'
    flip = True

    # General parameters
    height, width = 256, 256
    scale = 4

    # A* parameters
    cost_func = dorsal_cost_func
    allow_diagonal = False

    # Curvature parameters
    transpose_dims = False
    scales = np.array([0.04, 0.06, 0.08, 0.10], dtype=np.float32)

    # Curvature descriptors parameters
    curv_length = 1024
    uniform = False
    num_keypoints = 32
    feat_dim = 32

    # Build LNBNN index parameters
    index_fpaths = ['%.3f.ann' % s for s in scales]

    # Preprocessing
    print('Preprocessing')
    img = cv2.imread(fpath)
    resized_image, resized_mask, pre_transform =\
        F.preprocess_image(img, flip, height, width)

    # Localization
    print('Localization')
    imgs = [resized_image]
    masks = [resized_mask]
    layers = localization.build_model((None, 3, height, width))
    localization_weightsfile = join(
        'data', 'weights', 'weights_localization.pickle'
    )
    model.load_weights([
        layers['trans'], layers['loc']],
        localization_weightsfile
    )
    localization_func = theano_funcs.create_localization_infer_func(layers)

    localized_images, localized_masks, transforms =\
        F.localize(imgs, masks, height, width, localization_func)

    # Refinement
    print('Refinement')
    refined_localization, refined_mask = F.refine_localization(
        localized_images[0], flip, pre_transform, transforms[0],
        scale, height, width
    )

    # Segmentation
    print('Segmentation')
    segmentation_layers =\
        segmentation.build_model_batchnorm_full((None, 3, height, width))

    segmentation_weightsfile = join(
        'data', 'weights', 'weights_segmentation.pickle'
    )
    model.load_weights(segmentation_layers['seg_out'],
                       segmentation_weightsfile)
    segmentation_func = theano_funcs.create_segmentation_func(
        segmentation_layers)
    segmentations, refined_segmentations = F.segment_contour(
        [refined_localization], [refined_mask], scale, height, width,
        segmentation_func)

    # Keypoints
    print('Keypoints')
    start, end = F.find_keypoints(
        find_dorsal_keypoints, segmentations[0], localized_masks[0])

    # Extract Outline
    print('Extract Outline')
    outline = F.extract_outline(
        refined_localization, refined_mask, refined_segmentations[0], scale,
        start, end, cost_func, allow_diagonal)

    # Separate Edges
    print('Separate Edges')
    _, trailing_edge = F.separate_edges(
        separate_leading_trailing_edges, outline)

    # Compute Curvature
    print('Compute Curvature')
    curvature = F.compute_curvature(trailing_edge, scales, transpose_dims)

    # Compute Curvature Descriptors
    print('Compute Curvature Descriptors')
    feature_matrices = F.compute_curvature_descriptors(
        curvature, curv_length, scales, num_keypoints, uniform, feat_dim)

    # Build LNBNN Index
    print('Build LNBNN Index')
    for index_fpath, feature_matrix in zip(index_fpaths, feature_matrices):
        F.build_lnbnn_index(feature_matrix, index_fpath)


if __name__ == '__main__':
    example()
