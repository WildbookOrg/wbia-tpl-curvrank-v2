# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
import uuid

import wbia

from sklearn.model_selection import train_test_split
from os import makedirs
from os.path import join
from tqdm import tqdm


def prepare_images_with_contours(ibs, outdir, dbdir, check_aids=False):
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
        print('found %s bboxes with zero width/height; removing from training data' % bad_bboxes)
        bboxes = ut.compress(bboxes, bbox_is_valid)
        pids = ut.compress(pids, bbox_is_valid)
    print('after filtering bad bboxes we have %s pids' % len(pids))

    if check_aids:
        aids = ibs.get_part_annot_rowids(pids)
        annot_bboxes = ibs.get_annot_bboxes(aids)
        pids = [pid for pid, bbox in zip(pids, annot_bboxes) if bbox is not None]
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
    fpaths = [
        None if uri is None else join(dbdir, '_ibsdb/images', uri)
        for uri in ibs.get_image_uris(gids)
    ]

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


def prepare_training_data():
    dbdir = '/data/db'

    outdir = '/data/db/curvrank_training'
    makedirs(join(outdir), exist_ok=True)

    ibs = wbia.opendb(dbdir, allow_newdir=False)
    prepare_images_with_contours(ibs, outdir, dbdir, check_aids=True)


if __name__ == '__main__':
    prepare_training_data()
