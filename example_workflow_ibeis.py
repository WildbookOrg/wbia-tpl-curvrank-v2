from __future__ import absolute_import, division, print_function
from ibeis_curvrank import localization, model, segmentation, theano_funcs  # NOQA
from ibeis_curvrank.dorsal_utils import separate_leading_trailing_edges
from os.path import abspath, exists, join, split
import ibeis_curvrank.functional as F
import utool as ut
import numpy as np

from _plugin_depc import get_zipped

PATH = split(abspath(__file__))[0]

USE_DEPC = True

DEFAULT_WIDTH  = 256
DEFAULT_HEIGHT = 256
DEFAULT_SCALE  = 4

DEFAULT_CONFIG = {
    'curvrank_width'         : DEFAULT_WIDTH,
    'curvrank_height'        : DEFAULT_HEIGHT,
    'curvrank_scale'         : DEFAULT_SCALE,
    'localization_model_tag' : 'localization',
    'segmentation_model_tag' : 'segmentation',
    'outline_allow_diagonal' : False,
}


# images, list of np.ndarray: untouched input images.
# names, list: names of the individuals in images (one per image).
# flips, list: boolean indicating whether or not to L/R flip an image.
def pipeline(dataset_imageset_text, config=None):
    import ibeis
    from ibeis.init import sysres

    assert dataset_imageset_text in ['database', 'query']

    if config is None:
        config = DEFAULT_CONFIG

    # General parameters
    # height, width = 256, 256
    # scale = DEFAULT_SCALE

    # A* parameters
    # cost_func = dorsal_cost_func
    # allow_diagonal = False

    # Curvature parameters
    transpose_dims = False
    scales = np.array([0.04, 0.06, 0.08, 0.10], dtype=np.float32)

    # Curvature descriptors parameters
    curv_length = 1024
    uniform = False
    num_keypoints = 32
    feat_dim = 32

    # Because we don't have a dependency graph to ensure consistency, we need
    # to track failures manually.  When a pipeline stage fails, we set the
    # corresponding entry to False so that it is ignored downstream.

    ############################################################################

    dbdir = sysres.ensure_testdb_curvrank()
    ibs = ibeis.opendb(dbdir=dbdir)

    print('Preprocessing')
    imageset_rowid = ibs.get_imageset_imgsetids_from_text(dataset_imageset_text)
    gid_list = ibs.get_imageset_gids(imageset_rowid)

    aids_list = ibs.get_image_aids(gid_list)
    aid_list = ut.flatten(aids_list)
    names = ibs.get_annot_names(aid_list)

    success = len(gid_list) * [True]

    if USE_DEPC:
        resized_images = ibs.depc_image.get('preprocess', gid_list, 'resized_img',  config=config)
        resized_masks  = ibs.depc_image.get('preprocess', gid_list, 'mask_img',     config=config)
        pre_transforms = ibs.depc_image.get('preprocess', gid_list, 'pretransform', config=config)
    else:
        values = ibs.ibeis_plugin_curvrank_preprocessing(gid_list)
        resized_images, resized_masks, pre_transforms = values

    # Localization
    print('Localization')
    if USE_DEPC:
        localized_images = ibs.depc_image.get('localization', gid_list, 'localized_img', config=config)
        localized_masks  = ibs.depc_image.get('localization', gid_list, 'mask_img',  config=config)
        loc_transforms   = ibs.depc_image.get('localization', gid_list, 'transform', config=config)
    else:
        values = ibs.ibeis_plugin_curvrank_localization(resized_images, resized_masks)
        localized_images, localized_masks, loc_transforms = values

    # Refinement
    print('Refinement')
    if USE_DEPC:
        refined_localizations = ibs.depc_image.get('refinement', gid_list, 'refined_img', config=config)
        refined_masks         = ibs.depc_image.get('refinement', gid_list, 'mask_img',    config=config)
    else:
        values = ibs.ibeis_plugin_curvrank_refinement(gid_list, pre_transforms, loc_transforms)
        refined_localizations, refined_masks = values

    # Segmentation
    print('Segmentation')
    if USE_DEPC:
        segmentations          = ibs.depc_image.get('segmentation', gid_list, 'segmentations_img',         config=config)
        refined_segmentations  = ibs.depc_image.get('segmentation', gid_list, 'refined_segmentations_img', config=config)
    else:
        values = ibs.ibeis_plugin_curvrank_segmentation(refined_localizations, refined_masks)
        segmentations, refined_segmentations = values

    # NOTE: Tasks downstream from here may fail!  Need to check status.
    # Keypoints
    print('Keypoints')
    if USE_DEPC:
        success  = ibs.depc_image.get(      'keypoints', gid_list, 'success', config=config)
        starts = get_zipped(ibs.depc_image, 'keypoints', gid_list, 'start_y', 'start_x', config=config)
        ends   = get_zipped(ibs.depc_image, 'keypoints', gid_list, 'end_y',   'end_x',   config=config)
    else:
        values = ibs.ibeis_plugin_curvrank_keypoints(segmentations, localized_masks)
        success, starts, ends = values

    # Extract Outline
    print('Extract Outline')

    if USE_DEPC:
        success  = ibs.depc_image.get('outline', gid_list, 'success', config=config)
        outlines = ibs.depc_image.get('outline', gid_list, 'outline', config=config)
    else:
        args = success, starts, ends, refined_localizations, refined_masks, refined_segmentations
        success, outlines = ibs.ibeis_plugin_curvrank_outline(*args)

    # Separate Edges
    print('Separate Edges')
    trailing_edges = []
    for i, _ in enumerate(gid_list):
        if success[i]:
            _, trailing_edge = F.separate_edges(
                separate_leading_trailing_edges, outlines[i])
            if trailing_edge is None:
                success[i] = None
        else:
            trailing_edge = None
        trailing_edges.append(trailing_edge)

    # Compute Curvature
    print('Compute Curvature')
    curvatures = []
    for i, _ in enumerate(gid_list):
        if success[i]:
            curvature = F.compute_curvature(
                trailing_edges[i], scales, transpose_dims)
        else:
            curvature = None
        curvatures.append(curvature)

    # Compute Curvature Descriptors
    print('Compute Curvature Descriptors')
    feature_matrices_list = []
    for i, _ in enumerate(gid_list):
        if success[i]:
            feature_matrices = F.compute_curvature_descriptors(
                curvatures[i], curv_length, scales,
                num_keypoints, uniform, feat_dim)
        else:
            feature_matrices = None
        feature_matrices_list.append(feature_matrices)

    # Collect the images for which the pipeline was successful.
    valid_fmats, valid_names = [], []
    for i, _ in enumerate(gid_list):
        if success[i]:
            valid_fmats.append(feature_matrices_list[i])
            valid_names.append(names[i])

    # Aggregate the feature matrices.  Each descriptor is labeled with the
    # name of the individual from which it was taken, or None if unknown.
    lnbnn_data = {}
    fmats_by_scale = list(zip(*valid_fmats))
    for i, s in enumerate(scales):
        N = np.hstack([
            [name] * fmat.shape[0]
            for name, fmat in zip(valid_names, fmats_by_scale[i])
        ])
        D = np.vstack((fmats_by_scale[i]))

        lnbnn_data[s] = (D, N)
        assert D.shape[0] == N.shape[0], 'D.shape[0] != N.shape[0]'
        assert np.allclose(np.linalg.norm(D, axis=1), np.ones(D.shape[0]))

    return lnbnn_data


def example(output_path=None):
    assert exists(PATH)

    if output_path is None:
        import utool as ut
        output_path = abspath(join(PATH, '..', '_output'))
        ut.ensuredir(output_path)
        print('Using output_path=%r' % (output_path, ))

    print('Loading database images.')
    db_lnbnn_data = pipeline('database')

    # Build LNBNN index parameters.
    for s in db_lnbnn_data:
        # NOTE: This mem-mapped file must be persistent across queries!
        index_fpath = join(output_path, '%.3f.ann' % s)
        # Only need the descriptors to build the index.  The labels are
        # only used at inference time.
        D, _ = db_lnbnn_data[s]
        F.build_lnbnn_index(D, index_fpath)

    print('Loading query images for one encounter.')
    # Pass the query images through the pipeline.  It's okay not to know the
    # names, but all images must be of the same individual.
    qr_lnbnn_data = pipeline('query')

    # The neighbor index used to compute the norm. distance in LNBNN..
    k = 2
    # A name may appear in the database more than once, but we only want to
    # aggregate scores across all appearances of that name.
    # Run LNBNN identification for each scale independently and aggregate.
    agg_scores = {}
    for s in qr_lnbnn_data:
        index_fpath = join(output_path, '%.3f.ann' % s)
        # Need the descriptor labels from the database for a query.
        _, N = db_lnbnn_data[s]
        # Don't know the descriptor labels for a query.
        D, _ = qr_lnbnn_data[s]
        scores = F.lnbnn_identify(index_fpath, k, D, N)
        for name in scores:
            if name not in agg_scores:
                agg_scores[name] = 0.0
            agg_scores[name] += scores[name]

    unique_db_names = sorted(list(agg_scores.keys()))

    print('Results.')
    # More negative score => stronger evidence.
    idx = np.array([agg_scores[name] for name in unique_db_names]).argsort()
    ranked_db_names = [unique_db_names[i] for i in idx]
    ranked_scores = [agg_scores[name] for name in ranked_db_names]
    print(' Ranking: %s' % (', '.join(name for name in ranked_db_names)))
    print(' Scores: %s' % (', '.join('%.2f' % s for s in ranked_scores)))


if __name__ == '__main__':
    example()
