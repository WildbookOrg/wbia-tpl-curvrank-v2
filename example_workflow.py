from __future__ import absolute_import, division, print_function
from ibeis_curvrank import localization, model, segmentation, theano_funcs
from ibeis_curvrank.dorsal_utils import find_dorsal_keypoints, dorsal_cost_func
from ibeis_curvrank.dorsal_utils import separate_leading_trailing_edges
from os.path import isfile, join, abspath, split, exists
import ibeis_curvrank.functional as F
import numpy as np
import cv2


PATH = split(abspath(__file__))[0]


# images, list of np.ndarray: untouched input images.
# names, list: names of the individuals in images (one per image).
# flips, list: boolean indicating whether or not to L/R flip an image.
def pipeline(images, names, flips):
    import ibeis
    from ibeis.init import sysres

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

    # Because we don't have a dependency graph to ensure consistency, we need
    # to track failures manually.  When a pipeline stage fails, we set the
    # corresponding entry to False so that it is ignored downstream.
    success = len(images) * [True]

    ############################################################################

    dbdir = sysres.ensure_testdb_curvrank()
    ibs = ibeis.opendb(dbdir=dbdir)

    # Preprocessing
    # print('Preprocessing')
    # resized_images, resized_masks, pre_transforms = [], [], []
    # for i, _ in enumerate(images):
    #     resized_image, resized_mask, pre_transform =\
    #         F.preprocess_image(images[i], flips[i], height, width)

    #     resized_images.append(resized_image)
    #     resized_masks.append(resized_mask)
    #     pre_transforms.append(pre_transform)

    print('Preprocessing')
    imageset_rowid = ibs.get_imageset_imgsetids_from_text('database')
    gid_list = ibs.get_imageset_gids(imageset_rowid)
    values = ibs.ibeis_plugin_curvrank_preprocessing(gid_list, height=height, width=width)
    resized_images, resized_masks, pre_transforms = values

    # Localization
    print('Localization')
    layers = localization.build_model((None, 3, height, width))
    localization_weightsfile = join(PATH, '..', '_weights', 'weights_localization.pickle')
    model.load_weights([
        layers['trans'], layers['loc']],
        localization_weightsfile
    )
    localization_func = theano_funcs.create_localization_infer_func(layers)

    localized_images, localized_masks, loc_transforms =\
        F.localize(resized_images, resized_masks, height, width,
                   localization_func)

    # Refinement
    print('Refinement')
    refined_localizations, refined_masks = [], []
    for i, _ in enumerate(images):
        refined_localization, refined_mask = F.refine_localization(
            localized_images[i], flips[i],
            pre_transforms[i], loc_transforms[i],
            scale, height, width
        )

        refined_localizations.append(refined_localization)
        refined_masks.append(refined_mask)

    # Segmentation
    print('Segmentation')
    segmentation_layers =\
        segmentation.build_model_batchnorm_full((None, 3, height, width))

    segmentation_weightsfile = join(PATH, '..', '_weights', 'weights_segmentation.pickle')
    model.load_weights(segmentation_layers['seg_out'],
                       segmentation_weightsfile)
    segmentation_func = theano_funcs.create_segmentation_func(
        segmentation_layers)
    segmentations, refined_segmentations = F.segment_contour(
        refined_localizations, refined_masks, scale, height, width,
        segmentation_func)

    # NOTE: Tasks downstream from here may fail!  Need to check status.
    # Keypoints
    print('Keypoints')
    starts, ends = [], []
    for i, _ in enumerate(images):
        start, end = F.find_keypoints(
            find_dorsal_keypoints, segmentations[i], localized_masks[i])
        if start is None or end is None:
            success[i] = False

        starts.append(start)
        ends.append(end)

    # Extract Outline
    print('Extract Outline')
    outlines = []
    for i, _ in enumerate(images):
        if success[i]:
            outline = F.extract_outline(
                refined_localizations[i], refined_masks[i],
                refined_segmentations[i], scale,
                starts[i], ends[i], cost_func, allow_diagonal)
            if outline is None:
                success[i] = False
        else:
            outline = None
        outlines.append(outline)

    # Separate Edges
    print('Separate Edges')
    trailing_edges = []
    for i, _ in enumerate(images):
        if success[i]:
            _, trailing_edge = F.separate_edges(
                separate_leading_trailing_edges, outline)
            if trailing_edge is None:
                success[i] = None
        else:
            trailing_edge = None
        trailing_edges.append(trailing_edge)

    # Compute Curvature
    print('Compute Curvature')
    curvatures = []
    for i, _ in enumerate(images):
        if success[i]:
            curvature = F.compute_curvature(
                trailing_edges[i], scales, transpose_dims)
        else:
            curvature = None
        curvatures.append(curvature)

    # Compute Curvature Descriptors
    print('Compute Curvature Descriptors')
    feature_matrices_list = []
    for i, _ in enumerate(images):
        if success[i]:
            feature_matrices = F.compute_curvature_descriptors(
                curvature, curv_length, scales,
                num_keypoints, uniform, feat_dim)
        else:
            feature_matrices = None
        feature_matrices_list.append(feature_matrices)

    # Collect the images for which the pipeline was successful.
    valid_fmats, valid_names = [], []
    for i, _ in enumerate(images):
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

    return lnbnn_data


def example(output_path=None):
    assert exists(PATH)

    db_dir = join(PATH, '..', '_images', 'db')

    if output_path is None:
        import utool as ut
        output_path = abspath(join(PATH, '..', '_output'))
        ut.ensuredir(output_path)
        print('Using output_path=%r' % (output_path, ))

    assert exists(db_dir)
    db_fnames = [
        '17874.JPG', '17541.JPG',
        '23496.JPG', '25697.JPG',
        '26929.JPG', '27516.JPG'
    ]
    # CurvRank only handles left-view images.  The pipeline uses this to flip
    # right-view images.
    db_sides = ['Right', 'Right', 'Right', 'Left', 'Right', 'Right']
    qr_dir = join(PATH, '..', '_images', 'qr')
    assert exists(qr_dir)
    db_fpaths = [join(db_dir, f) for f in db_fnames]
    # The names corresponding to the images in the database.
    db_names = ['F272', 'F272', 'F274', 'F274', 'F276', 'F276']

    # Two images from the same encounter to be used as a query.
    qr_fnames = ['23441.JPG', '23442.JPG']
    qr_sides = ['Right', 'Left']
    qr_fpaths = [join(qr_dir, f) for f in qr_fnames]
    # The individual is F272, but pretend we don't know it for the example.
    qr_names = [None, None]

    print('Loading database images.')
    db_images = [cv2.imread(db_fpath) for db_fpath in db_fpaths]
    db_flips = [True if side == 'Right' else False for side in db_sides]

    # Pass the database images through the pipeline.
    db_lnbnn_data = pipeline(db_images, db_names, db_flips)

    # Build LNBNN index parameters.
    for s in db_lnbnn_data:
        # NOTE: This mem-mapped file must be persistent across queries!
        index_fpath = join(output_path, '%.3f.ann' % s)
        if not isfile(index_fpath):
            # Only need the descriptors to build the index.  The labels are
            # only used at inference time.
            D, _ = db_lnbnn_data[s]
            F.build_lnbnn_index(D, index_fpath)

    print('Loading query images for one encounter.')
    qr_images = [cv2.imread(qr_fpath) for qr_fpath in qr_fpaths]
    qr_flips = [True if side == 'Right' else False for side in qr_sides]

    # Pass the query images through the pipeline.  It's okay not to know the
    # names, but all images must be of the same individual.
    qr_lnbnn_data = pipeline(qr_images, qr_names, qr_flips)

    # The neighbor index used to compute the norm. distance in LNBNN..
    k = 2
    # A name may appear in the database more than once, but we only want to
    # aggregate scores across all appearances of that name.
    unique_db_names = np.unique(db_names)
    agg_scores = {name: 0.0 for name in unique_db_names}
    # Run LNBNN identification for each scale independently and aggregate.
    for s in qr_lnbnn_data:
        index_fpath = join(output_path, '%.3f.ann' % s)
        # Need the descriptor labels from the database for a query.
        _, N = db_lnbnn_data[s]
        # Don't know the descriptor labels for a query.
        D, _ = qr_lnbnn_data[s]
        scores = F.lnbnn_identify(index_fpath, k, D, N)
        for name in scores:
            agg_scores[name] += scores[name]

    print('Results.')
    # More negative score => stronger evidence.
    idx = np.array([agg_scores[name] for name in unique_db_names]).argsort()
    ranked_db_names = [unique_db_names[i] for i in idx]
    ranked_scores = [agg_scores[name] for name in ranked_db_names]
    print(' Ranking: %s' % (', '.join(name for name in ranked_db_names)))
    print(' Scores: %s' % (', '.join('%.2f' % s for s in ranked_scores)))


if __name__ == '__main__':
    example()
