import affine
import cv2
import cPickle as pickle
import numpy as np
import dorsal_utils
import imutils
import matplotlib
matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt


def preprocess_images_star(fpath_side, imsize, output_targets):
    return preprocess_images(
        *fpath_side,
        imsize=imsize, output_targets=output_targets
    )


def preprocess_images(fpath, side, imsize, output_targets):
    resz_target = output_targets[fpath]['resized']
    trns_target = output_targets[fpath]['transform']

    img = cv2.imread(fpath)
    # mirror images marked as "Right" to simulate a left-view
    if side.lower() == 'right':
        img = img[:, ::-1, :]

    resz, M = imutils.center_pad_with_transform(img, imsize)
    _, resz_buf = cv2.imencode('.png', resz)

    with resz_target.open('wb') as f1,\
            trns_target.open('wb') as f2:
        f1.write(resz_buf)
        pickle.dump(M, f2, pickle.HIGHEST_PROTOCOL)


# input1_targets: localization_targets
# input2_targets: segmentation_targets
def find_keypoints(fpath, input1_targets, input2_targets, output_targets):
    coords_target = output_targets[fpath]['keypoints-coords']
    visual_target = output_targets[fpath]['keypoints-visual']

    loc_fpath = input1_targets[fpath]['localization'].path
    loc = cv2.imread(loc_fpath)

    seg_fpath = input2_targets[fpath]['segmentation-data'].path
    with open(seg_fpath, 'rb') as f:
        seg = pickle.load(f)

    start, end = dorsal_utils.find_keypoints(seg[:, :, 0])

    # TODO: what to write for failed extractions?
    if start is not None:
        cv2.circle(loc, tuple(start[::-1]), 3, (255, 0, 0), -1)
    if end is not None:
        cv2.circle(loc, tuple(end[::-1]), 3, (0, 0, 255), -1)
    _, visual_buf = cv2.imencode('.png', loc)
    with coords_target.open('wb') as f1,\
            visual_target.open('wb') as f2:
        pickle.dump((start, end), f1, pickle.HIGHEST_PROTOCOL)
        f2.write(visual_buf)


# input1_targets: localization_targets
# input2_targets: segmentation_targets
# input3_targets: keypoints_targets
def extract_outline(fpath, scale,
                    input1_targets, input2_targets, input3_targets,
                    output_targets):
    coords_target = output_targets[fpath]['outline-coords']
    visual_target = output_targets[fpath]['outline-visual']

    loc_fpath = input1_targets[fpath]['localization-full'].path
    loc = cv2.imread(loc_fpath)

    seg_fpath = input2_targets[fpath]['segmentation-full-data']
    key_fpath = input3_targets[fpath]['keypoints-coords']
    with seg_fpath.open('rb') as f1,\
            key_fpath.open('rb') as f2:
        segm = pickle.load(f1)
        (start, end) = pickle.load(f2)

    if start is not None and end is not None:
        Mscale = affine.build_scale_matrix(scale)
        points_orig = np.vstack((start, end))[:, ::-1]  # ij -> xy
        points_refn = affine.transform_points(Mscale, points_orig)

        start_refn, end_refn = np.floor(points_refn[:, ::-1]).astype(np.int32)
        outline = dorsal_utils.extract_outline(loc, segm, start_refn, end_refn)
    else:
        outline = np.array([])

    # TODO: what to write for failed extractions?
    if outline.shape[0] > 0:
        loc[outline[:, 0], outline[:, 1]] = (255, 0, 0)

    _, visual_buf = cv2.imencode('.png', loc)
    with coords_target.open('wb') as f1,\
            visual_target.open('wb') as f2:
        pickle.dump(outline, f1, pickle.HIGHEST_PROTOCOL)
        f2.write(visual_buf)


#input1_targets: localization_targets
#input2_targets: extract_outline_targets
def separate_edges(fpath, input1_targets, input2_targets, output_targets):
    localization_target = input1_targets[fpath]['localization-full']
    outline_coords_target = input2_targets[fpath]['outline-coords']

    loc = cv2.imread(localization_target.path)
    with open(outline_coords_target.path, 'rb') as f:
        outline = pickle.load(f)

    # no successful outline could be found
    if outline.shape[0] > 0:
        idx = dorsal_utils.separate_leading_trailing_edges(outline)
        if idx is not None:
            leading_edge = outline[:idx]
            trailing_edge = outline[idx:]

            loc[leading_edge[:, 0], leading_edge[:, 1]] = (255, 0, 0)
            loc[trailing_edge[:, 0], trailing_edge[:, 1]] = (0, 0, 255)
        else:
            leading_edge, trailing_edge = None, None
    else:
        leading_edge, trailing_edge = None, None

    vis_target = output_targets[fpath]['visual']
    _, loc_buf = cv2.imencode('.png', loc)

    with vis_target.open('wb') as f1:
        f1.write(loc_buf)

    leading_target = output_targets[fpath]['leading-coords']
    trailing_target = output_targets[fpath]['trailing-coords']
    with leading_target.open('wb') as f1,\
            trailing_target.open('wb') as f2:
        pickle.dump(leading_edge, f1, pickle.HIGHEST_PROTOCOL)
        pickle.dump(trailing_edge, f2, pickle.HIGHEST_PROTOCOL)


def compute_curvature_star(fpath_scales, oriented,
                           input_targets, output_targets):
    return compute_curvature(
        *fpath_scales,
        oriented=oriented,
        input_targets=input_targets, output_targets=output_targets
    )


#input_targets: extract_high_resolution_outline_targets
def compute_curvature(fpath, scales, oriented,
                      input_targets, output_targets):
    trailing_coords_target = input_targets[fpath]['trailing-coords']
    with open(trailing_coords_target.path, 'rb') as f:
        trailing_edge = pickle.load(f)

    if trailing_edge is not None:
        # compute_curvature uses (x, y) coordinates
        trailing_edge = trailing_edge[:, ::-1]
        if oriented:
            curv = dorsal_utils.oriented_curvature(trailing_edge, scales)
        else:
            curv = dorsal_utils.block_curvature(trailing_edge, scales)
    # write the failures too or it seems like the task did not complete
    else:
        curv = None

    curv_target = output_targets[fpath]['curvature']
    with curv_target.open('a') as h5f:
        # store each scale (column) of the curvature matrix separately
        for j, scale in enumerate(scales):
            if curv is not None:
                h5f.create_dataset('%.3f' % scale, data=curv[:, j])
            else:
                h5f.create_dataset('%.3f' % scale, data=None, dtype=np.float32)


def visualize_individuals(fpath, input_targets, output_targets):
    separate_edges_target = input_targets[fpath]['visual']
    img = cv2.imread(separate_edges_target.path)

    visualization_target = output_targets[fpath]['image']
    _, img_buf = cv2.imencode('.png', img)
    with visualization_target.open('wb') as f:
        f.write(img_buf)


def identify_encounters(qind, qr_curv_dict, db_curv_dict, simfunc,
                        output_targets):
    qencs = qr_curv_dict[qind].keys()
    dindivs = db_curv_dict.keys()
    assert qencs, 'empty encounter list for %s' % qind
    for qenc in qencs:
        result_dict = {}
        qcurvs = qr_curv_dict[qind][qenc]
        for dind in dindivs:
            dcurvs = db_curv_dict[dind]
            # mxn matrix: m query curvs, n db curvs for an individual
            S = np.zeros((len(qcurvs), len(dcurvs)), dtype=np.float32)
            for i, qcurv in enumerate(qcurvs):
                for j, dcurv in enumerate(dcurvs):
                    S[i, j] = simfunc(qcurv, dcurv)

            result_dict[dind] = S

        with output_targets[qind][qenc].open('wb') as f:
            pickle.dump(result_dict, f, pickle.HIGHEST_PROTOCOL)


# input1_targets: evaluation_targets (the result dicts)
# input2_targets: edges_targets (the separate_edges visualizations)
# input3_targets: block_curv_targets (the curvature vectors)
def visualize_misidentifications(qind, qr_dict, db_dict, num_db, num_qr,
                                 input1_targets, input2_targets,
                                 input3_targets, output_targets):
    dindivs = np.hstack(db_dict.keys())  # TODO: add sorted() everywhere
    qencs = input1_targets[qind].keys()
    for qenc in qencs:
        with input1_targets[qind][qenc].open('rb') as f:
            result_dict = pickle.load(f)
        result_across_db = np.hstack([result_dict[dind] for dind in dindivs])
        indivs_across_db = np.hstack(
            [np.repeat(dind, len(db_dict[dind])) for dind in dindivs]
        )
        db_fnames = np.hstack([db_dict[dind] for dind in dindivs])
        query_fnames = np.hstack(qr_dict[qind][qenc])

        assert db_fnames.shape[0] == result_across_db.shape[1]
        best_score_per_query = result_across_db.min(axis=1)
        qr_best_idx = best_score_per_query.argsort(axis=0)[0:num_qr]
        qr_best_fnames = query_fnames[qr_best_idx]
        qr_best_scores = result_across_db[qr_best_idx]

        db_best_idx = qr_best_scores.argsort(axis=1)

        db_best_fnames = db_fnames[db_best_idx[:, 0:num_db]]
        db_best_scores = np.array([
            qr_best_scores[i, db_best_idx[i]]
            for i in np.arange(db_best_idx.shape[0])
        ])
        db_best_indivs = indivs_across_db[db_best_idx]

        db_best_qr_idx = np.argmax(db_best_indivs == qind, axis=1)
        db_best_qr_fnames = db_fnames[db_best_idx][
            np.arange(db_best_qr_idx.shape[0]), db_best_qr_idx
        ]

        db_best_qr_indivs = db_best_indivs[
            np.arange(db_best_idx.shape[0]), db_best_qr_idx
        ]

        db_best_qr_scores = db_best_scores[
            np.arange(db_best_qr_idx.shape[0]), db_best_qr_idx
        ]

        f, axarr = plt.subplots(
            2 + min(db_best_fnames.shape[1], num_db),  # rows
            min(qr_best_fnames.shape[0], num_qr),      # cols
            figsize=(22., 12.)
        )
        if axarr.ndim == 1:
            axarr = np.expand_dims(axarr, axis=1)  # ensure 2d
        db_rows = []
        for i, _ in enumerate(qr_best_fnames):
            qr_edge_fname = input2_targets[qr_best_fnames[i]]['visual']
            qr_curv_fname = input3_targets[qr_best_fnames[i]]['curvature']
            db_edge_fnames = [
                input2_targets[name]['visual'] for name in db_best_fnames[i]
            ]
            db_qr_edge_fname = input2_targets[db_best_qr_fnames[i]]['visual']
            db_qr_curv_fname = input3_targets[
                db_best_qr_fnames[i]
            ]['curvature']

            db_curv_fnames = [
                input3_targets[name]['curvature'] for name in db_best_fnames[i]
            ]

            qr_img = cv2.resize(cv2.imread(qr_edge_fname.path), (256, 256))
            cv2.putText(
                qr_img, '%s: %s' % (qind, qenc),
                (10, qr_img.shape[0] - 10),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0)
            )

            db_row = []
            for didx, db_edge_fname in enumerate(db_edge_fnames):
                db_img = cv2.resize(cv2.imread(db_edge_fname.path), (256, 256))
                dind = db_best_indivs[i, didx]
                dscore = db_best_scores[i, didx]
                cv2.putText(
                    db_img, '%d) %s: %.6f' % (
                        1 + didx, db_best_indivs[i, didx], dscore),
                    (10, db_img.shape[0] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.0,
                    (0, 255, 0) if dind == qind else (0, 0, 255)
                )
                db_row.append(db_img)

            db_qr_img = cv2.resize(
                cv2.imread(db_qr_edge_fname.path), (256, 256),
            )
            cv2.putText(
                db_qr_img, '%d) %s: %.6f' % (
                    1 + db_best_qr_idx[i], db_best_qr_indivs[i],
                    db_best_qr_scores[i]),
                (10, db_qr_img.shape[0] - 10),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0)
            )

            db_row = np.hstack(db_row)
            db_rows.append(np.hstack((qr_img, db_row, db_qr_img)))

            with qr_curv_fname.open('rb') as f:
                qcurv = pickle.load(f)
            axarr[0, i].set_title('%s: %s' % (qind, qenc), size='xx-small')
            axarr[0, i].plot(np.arange(qcurv.shape[0]), qcurv)
            axarr[0, i].set_ylim((0, 1))
            axarr[0, i].set_xlim((0, qcurv.shape[0]))
            axarr[0, i].xaxis.set_visible(False)
            for didx, db_curv_fname in enumerate(db_curv_fnames, start=1):
                with db_curv_fname.open('rb') as f:
                    dcurv = pickle.load(f)
                axarr[didx, i].plot(np.arange(dcurv.shape[0]), dcurv)
                axarr[didx, i].set_title(
                    '%d) %s: %.6f' % (
                        didx, db_best_indivs[i, didx - 1],
                        db_best_scores[i, didx - 1]),
                    size='xx-small')
                axarr[didx, i].set_ylim((0, 1))
                axarr[didx, i].set_xlim((0, dcurv.shape[0]))
                axarr[didx, i].xaxis.set_visible(False)

            with db_qr_curv_fname.open('rb') as f:
                db_qr_curv = pickle.load(f)
            axarr[-1, i].plot(np.arange(db_qr_curv.shape[0]), db_qr_curv)
            axarr[-1, i].set_title(
                '%d) %s: %.6f' % (
                    1 + db_best_qr_idx[i],
                    db_best_qr_indivs[i],
                    db_best_qr_scores[i]),
                size='xx-small')
            axarr[-1, i].set_ylim((0, 1))
            axarr[-1, i].set_xlim((0, db_qr_curv.shape[0]))
            axarr[-1, i].xaxis.set_visible(False)

        grid = np.vstack(db_rows)

        _, edges_buf = cv2.imencode('.png', grid)
        with output_targets[qind][qenc]['separate-edges'].open('wb') as f:
            f.write(edges_buf)

        with output_targets[qind][qenc]['curvature'].open('wb') as f:
            plt.savefig(f, bbox_inches='tight')
        plt.clf()
        plt.close()
