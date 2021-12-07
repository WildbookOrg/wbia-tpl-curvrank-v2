# -*- coding: utf-8 -*-
import numpy as np
import utool as ut
import pandas as pd
import pickle
import uuid

import wbia

from sklearn.model_selection import train_test_split
from os import makedirs, path
from os.path import join
from tqdm import tqdm


def prepare_images_with_contours(ibs, outdir, dbdir, check_aids=True, mirror_rights=True):
    all_pids = ibs.get_valid_part_rowids()
    all_contour_dicts = ibs.get_part_contour(all_pids)
    pids_contours = [
        (pid, contour_dict)
        for pid, contour_dict in zip(all_pids, all_contour_dicts)
        if contour_dict and contour_dict['contour']
    ]

    # the contours constitute the training data
    print(
        'Found %d parts and %d parts with contours.' % (len(all_pids), len(pids_contours))
    )

    pids, contour_dicts = zip(*pids_contours)

    bboxes = ibs.get_part_bboxes(pids)

    # we have a few 0 width/height parts that we have to clean
    bbox_is_valid = [bb[2] > 5 and bb[3] > 5 for bb in bboxes]
    bad_bboxes = bbox_is_valid.count(False)
    if bad_bboxes > 0:
        print(
            'found %s bboxes with zero width/height; removing from training data'
            % bad_bboxes
        )
        bboxes = ut.compress(bboxes, bbox_is_valid)
        pids = ut.compress(pids, bbox_is_valid)
    print('after filtering bad bboxes we have %s pids' % len(pids))

    if check_aids:
        aids = ibs.get_part_annot_rowids(pids)
        annot_bboxes = ibs.get_annot_bboxes(aids)
        pids = [
            pid
            for pid, bbox in zip(pids, annot_bboxes)
            if bbox is not None and bbox[2] != 0 and bbox[3] != 0
        ]
        bboxes = ibs.get_part_bboxes(pids)
    print('after filtering None annot bboxes we have %s pids' % len(pids))

    aids = ibs.get_part_annot_rowids(pids)

    names = ibs.get_annot_names(aids)
    # Generate unique names for those individuals where we don't know the name.
    unique_names = [name if name != '____' else str(uuid.uuid4()) for name in names]
    names_to_split = list(set(unique_names))
    train_names, valid_names = train_test_split(names_to_split, train_size=0.8)
    train_names, valid_names = set(train_names), set(valid_names)
    assert not (train_names & valid_names), train_names & valid_names

    gids = ibs.get_annot_gids(aids)
    viewpoints = ibs.get_annot_viewpoints(aids)
    imwidths = ibs.get_image_widths(gids)

    # mirroring logic
    if mirror_rights:
        print('Mirroring right images')
        fpaths = left_mirrofied_image_fpaths(ibs, gids, viewpoints, dbdir)
        bboxes = left_mirrofied_bboxes(bboxes, viewpoints, imwidths)
        contour_dicts = left_mirrofied_contour_dicts(contour_dicts, viewpoints)
    else:
        fpaths = [
            None if uri is None else join(dbdir, '_ibsdb/images', uri)
            for uri in ibs.get_image_uris(gids)
        ]

    contour_dicts = ensure_l_to_r_contours(contour_dicts)

    contours_dir = join(outdir, 'contours')
    makedirs(contours_dir, exist_ok=True)

    data_list = []
    # tqdm is just a wrapper on an enumeration that shows a progress bar
    tqdm_iter = tqdm(
        enumerate(zip(contour_dicts, unique_names, bboxes, fpaths)), total=len(fpaths)
    )
    for i, (contour_dict, name, bbox, fpath) in tqdm_iter:
        contour_data = np.vstack(
            [(c['x'], c['y']) for c in contour_dict['contour']['segment']]
        )
        radii = (
            np.hstack([c['r'] * bbox[2] for c in contour_dict['contour']['segment']])
            / 2.0
        )  # TODO: just a test

        occluded = np.hstack([c['flag'] for c in contour_dict['contour']['segment']])

        pts_xy = contour_data * np.array(bbox[2:4]) + np.array(bbox[0:2])

        contour_fname = '%d.pickle' % i
        contour_fpath = join(contours_dir, '%s' % contour_fname)
        data = {'contour': pts_xy, 'radii': radii, 'occluded': occluded}

        with open(contour_fpath, 'wb') as f:
            pickle.dump(data, f, 4)

        is_train = '%d' % (name in train_names)

        data_list.append((fpath, name, contour_fpath, is_train))

    df = pd.DataFrame(data_list)
    df.columns = ['Image', 'Name', 'Contour', 'Train']
    # imagepath, name, contour pickle path, boolean on if row represents train(1) or test(0) data
    df.to_csv(join(outdir, 'train.csv'), index=False)


# def check_mirroring(ibs, pids, stop_after=20, target_dir='/tmp/check_mirroring'):
#     all_pids = ibs.get_valid_part_rowids()
#     all_contour_dicts = ibs.get_part_contour(all_pids)
#     pids_contours = [
#         (pid, contour_dict)
#         for pid, contour_dict in zip(all_pids, all_contour_dicts)
#         if contour_dict and contour_dict['contour']
#     ]

#     # the contours constitute the training data
#     print(
#         'Found %d parts and %d parts with contours.' % (len(all_pids), len(pids_contours))
#     )

#     pids, contour_dicts = zip(*pids_contours)

#     bboxes = ibs.get_part_bboxes(pids)

#     # we have a few 0 width/height parts that we have to clean
#     bbox_is_valid = [bb[2] > 5 and bb[3] > 5 for bb in bboxes]
#     bad_bboxes = bbox_is_valid.count(False)
#     if bad_bboxes > 0:
#         print('found %s bboxes with zero width/height; removing from training data' % bad_bboxes)
#         bboxes = ut.compress(bboxes, bbox_is_valid)
#         pids = ut.compress(pids, bbox_is_valid)
#     print('after filtering bad bboxes we have %s pids' % len(pids))

#     if check_aids:
#         aids = ibs.get_part_annot_rowids(pids)
#         annot_bboxes = ibs.get_annot_bboxes(aids)
#         pids = [pid for pid, bbox in zip(pids, annot_bboxes) if bbox is not None]
#         bboxes = ibs.get_part_bboxes(pids)
#     print('after filtering None annot bboxes we have %s pids' % len(pids))

#     aids = ibs.get_part_annot_rowids(pids)
#     viewpoints = ibs.get_annot_viewpoints(aids)
#     is_right = [view == 'right' for view in viewpoints]

#     #filter only rights
#     pids = ut.compress(pids, is_right)[:stop_after]
#     aids = ut.compress(pids, is_right)[:stop_after]
#     viewpoints = ut.compress(viewpoints, is_right)[:stop_after]

#     gids = ibs.get_annot_gids(aids)
#     imwidths = ibs.get_image_widths(gids)

#     fpaths = left_mirrofied_image_fpaths(ibs, gids, viewpoints, dbdir = '/data/db')
#     bboxes = left_mirrofied_bboxes(bboxes, viewpoints, imwidths)
#     contour_dicts = left_mirrofied_contour_dicts(contour_dicts, viewpoints)

#     from PIL import Image
#     for (fpath, bbox, contour, pid) in zip(fpaths, bboxes, contour_dicts, pids):
#         with Image.open(fpath) as im:


def prepare_training_datams():
    dbdir = '/data/db'

    outdir = '/data/db/curvrank_training'
    makedirs(join(outdir), exist_ok=True)

    ibs = wbia.opendb(dbdir, allow_newdir=False)
    prepare_images_with_contours(ibs, outdir, dbdir, check_aids=True)


# right/left mirroring is already integrated into our deployed pipeline. CurvRank training runs off of a .csv reference file and the filesystem rather than the wbia database. So, right/left mirroring during training is just done once while making this .csv and associated files.
def left_mirrofied_image_fpaths(ibs, gid_list, viewpoint_list, dbdir):
    # slower but simpler code if we just do this as a for loop
    unflipped_fpaths = [
        None if uri is None else join(dbdir, '_ibsdb/images', uri)
        for uri in ibs.get_image_uris(gid_list)
    ]

    fpaths = []
    import cv2

    tqdm_iter = tqdm(zip(gid_list, viewpoint_list, unflipped_fpaths))
    for gid, viewpoint, orig_fp in tqdm_iter:
        if viewpoint is None or 'right' not in viewpoint:
            fpaths.append(orig_fp)
        else:  # flip right image
            flipped_fpath = (
                path.splitext(orig_fp)[0] + '-mirrored' + path.splitext(orig_fp)[1]
            )
            if not path.exists(flipped_fpath):
                # flip and save image to disk if it isn't there already
                image_arr = ibs.get_images(gid)
                flimage_arr = cv2.flip(image_arr, 1)
                cv2.imwrite(flipped_fpath, flimage_arr)
            fpaths.append(flipped_fpath)
    return fpaths


def left_mirrofied_bboxes(bboxes, viewpoints, imwidths):
    mirrofied_bboxes = [
        [imwidth - (bbox[0] + bbox[2]), bbox[1], bbox[2], bbox[3]]
        if viewpoint is not None and 'right' in viewpoint
        else bbox
        for bbox, viewpoint, imwidth in zip(bboxes, viewpoints, imwidths)
    ]
    return mirrofied_bboxes


def left_mirrofied_contour_dicts(contour_dicts, viewpoints):
    mirrofied_contour_dicts = [
        mirror_contours(contour_dict)
        if viewpoint is not None and 'right' in viewpoint
        else contour_dict
        for contour_dict, viewpoint in zip(contour_dicts, viewpoints)
    ]
    return mirrofied_contour_dicts


def ensure_l_to_r_contours(contour_dicts):
    return [l_to_r_contour(contour_dict) for contour_dict in contour_dicts]


# ensures that all contours go left to right
def l_to_r_contour(contour_dict):
    contour = contour_dict['contour']
    segment = contour['segment']
    begin_idx = contour['begin']
    end_idx = contour['end']
    start_x = segment[begin_idx]['x']
    end_x = segment[end_idx]['x']
    if start_x > end_x:
        new_segment = segment[::-1]
        new_start_idx = len(segment) - end_idx - 1
        new_end_idx = len(segment) - begin_idx - 1
        contour = {'begin': new_start_idx, 'end': new_end_idx, 'segment': new_segment}
        contour_dict = {'contour': contour}
    return contour_dict


def mirror_contours(contour):
    ctour = contour['contour']
    segment = ctour['segment']
    mirrored_seg = [
        {'x': 1 - p['x'], 'y': p['y'], 'r': p['r'], 'flag': p['flag']} for p in segment
    ]
    new_contour = {
        'begin': ctour['begin'],
        'end': ctour['end'],
        'segment': mirrored_seg,
    }
    cdict = {'contour': new_contour}
    return cdict


if __name__ == '__main__':
    dbdir = '/data/db'

    outdir = '/data/db/curvrank_training'
    makedirs(join(outdir), exist_ok=True)

    ibs = wbia.opendb(dbdir, allow_newdir=False)
    prepare_images_with_contours(ibs, outdir, dbdir, check_aids=True)
