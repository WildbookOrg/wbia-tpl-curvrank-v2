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


def prepare_images_with_contours(ibs, outdir, dbdir):
    all_pids = ibs.get_valid_part_rowids()
    all_contour_dicts = ibs.get_part_contour(all_pids)
    pids_contours = [
        (pid, contour_dict)
        for pid, contour_dict in zip(all_pids, all_contour_dicts)
        if contour_dict and contour_dict['contour']
    ]

    print(
        'Found %d parts and %d parts with contours.' % (len(all_pids), len(pids_contours))
    )

    pids, contour_dicts = zip(*pids_contours)

    aids = ibs.get_part_annot_rowids(pids)
    names = ibs.get_annot_names(aids)
    # Generate unique names for those individuals where we don't know the name.
    unique_names = [name if name != '____' else str(uuid.uuid4()) for name in names]
    names_to_split = list(set(unique_names))
    train_names, valid_names = train_test_split(names_to_split, train_size=0.8)
    train_names, valid_names = set(train_names), set(valid_names)
    assert not (train_names & valid_names), train_names & valid_names

    bboxes = ibs.get_part_bboxes(pids)

    gids = ibs.get_annot_gids(aids)
    fpaths = [
        None if uri is None else join(dbdir, '_ibsdb/images', uri)
        for uri in ibs.get_image_uris(gids)
    ]

    data_list = []
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
        contour_fpath = join(outdir, 'contours', '%s' % contour_fname)
        data = {'contour': pts_xy, 'radii': radii, 'occluded': occluded}

        with open(contour_fpath, 'wb') as f:
            pickle.dump(data, f, 4)

        is_train = '%d' % (name in train_names)

        data_list.append((fpath, name, contour_fpath, is_train))

    df = pd.DataFrame(data_list)
    df.columns = ['Image', 'Name', 'Contour', 'Train']
    df.to_csv(join(outdir, 'train.csv'), index=False)


def prepare_training_data():
    dbdir = '/home/mankow/Research/databases/flukebook'

    outdir = 'data'
    makedirs(join(outdir), exist_ok=True)

    ibs = wbia.opendb(dbdir, allow_newdir=False)
    prepare_images_with_contours(ibs, outdir, dbdir)


if __name__ == '__main__':
    prepare_training_data()
