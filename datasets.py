import cv2
import numpy as np
import sqlite3
from os import listdir
from os.path import basename, isdir, isfile, join, splitext


def load_dataset(name):
    if name == 'nz':
        return load_nz_dataset()
    elif name == 'sdrp':
        return load_sdrp_dataset([2013, 2014, 2015, 2016])
    else:
        assert False, 'bad dataset name: %s' % (name)


def load_nz_dataset():
    data_dir = '/media/hdd/hendrik/datasets/nz-dolphins'
    individuals = listdir(data_dir)
    data_list = []
    for indiv_name in individuals:
        indiv_dir = join(data_dir, indiv_name)
        # encounters have "done" as suffix
        encounters = [
            enc for enc in listdir(indiv_dir)
            if enc.lower().endswith('done') and not
            enc.startswith('.')
        ]
        for enc_name in encounters:
            enc_dir = join(indiv_dir, enc_name)
            images = listdir(enc_dir)
            for img_name in images:
                img_fpath = join(enc_dir, img_name)
                img = cv2.imread(img_fpath)
                if img is not None:
                    data_list.append((
                        img_fpath,
                        indiv_name,
                        enc_name,
                    ))

    return data_list


def load_sdrp_dataset(years):
    database = '/home/hendrik/projects/sdrp/databases/sdrp.db'
    conn = sqlite3.connect(database)
    cur = conn.cursor()

    # inner join because we don't want sightings that are not in the catalog
    cur.execute(
        'select '
        '  SurveyNum,'
        '  Sighting,'
        '  Date, Alias, Image, ImageSide from IndivSighting '
        'inner join Sighting on '
        '  IndivSighting.SightingFID = Sighting.ID '
        'inner join Individuals on '
        '  IndivSighting.IndividualsFID = Individuals.ID '
        'inner join IndivImage on '
        '  IndivImage.IndivSightingFID = IndivSighting.ID '
        'inner join Survey on '
        '  Sighting.SurveyFID = Survey.ID'
    )

    # keep only images from individuals that are in the catalog
    catalog_aliases = set()
    for (_, _, _, alias, _, _) in cur.fetchall():
        catalog_aliases.add(str(alias))

    data_dir = '/media/sdrp/SDRP Data'
    sighting_dir_list = []
    for year in years:
        sightings_dir = join(data_dir, '%s Dig Pics' % (year))
        sighting_dirs = listdir(sightings_dir)
        for sighting_dir in sighting_dirs:
            sighting_dir_list.append(
                join(data_dir, sightings_dir, sighting_dir)
            )

    data_list = []
    for sighting_dir in sighting_dir_list:
        if not isdir(sighting_dir):
            continue
        fname_list = listdir(sighting_dir)
        for fname in fname_list:
            try:
                alias, _, survey, sighting, date, _ = fname.strip().split(' ')
            except ValueError:
                continue

            if survey.lower().startswith('s'):
                survey = survey[1:]
            if sighting.lower().startswith('s'):
                sighting = sighting[1:]

            image_filepath = join(sighting_dir, fname)
            assert isfile(image_filepath), 'not a file: %s' % (image_filepath)
            # NOTE: we only use left-view images for the moment
            # we remove individuals that are not in the catalog to get rid of
            # "possible"-tagged images, e.g., "DUposs6512"
            if '(L)' in fname and alias in catalog_aliases:
                data_list.append((
                    image_filepath,
                    '%s-%s' % (survey, sighting),
                    alias,
                ))

    return data_list


def separate_database_queries(name, fpath_list, ind_list, enc_list, curv_dict):
    if name == 'nz':
        return separate_nz_dataset(fpath_list, ind_list, enc_list, curv_dict)
        #return separate_sdrp_dataset(fpath_list, ind_list, enc_list, curv_dict)
    elif name == 'sdrp':
        return separate_sdrp_dataset()
    else:
        assert False, 'bad dataset name: %s' % (name)


def separate_nz_dataset(fpath_list, ind_list, enc_list, curv_dict):
    ind_enc_curv_dict = {}
    for fpath, ind, enc in zip(fpath_list, ind_list, enc_list):
        fname = splitext(basename(fpath))[0]
        if fname in curv_dict.keys():
            if ind not in ind_enc_curv_dict:
                ind_enc_curv_dict[ind] = {}
            if enc not in ind_enc_curv_dict[ind]:
                ind_enc_curv_dict[ind][enc] = []
            ind_enc_curv_dict[ind][enc].append(curv_dict[fname])

    # db_dict: {'i1': [v1, v2, ..., vn]}
    # qr_dict: {'i2': {'e1': [v1, v2, ..., vm]}}
    db_dict, qr_dict = {}, {}
    individuals = ind_enc_curv_dict.keys()
    for ind in individuals:
        encounters = ind_enc_curv_dict[ind].keys()
        num_encounters = len(encounters)
        if num_encounters > 1:
            num_curvs = [
                len(ind_enc_curv_dict[ind][enc]) for enc in encounters
            ]
            max_idx = np.argmax(num_curvs)
            for idx, enc in enumerate(encounters):
                if idx == max_idx:
                    db_dict[ind] = ind_enc_curv_dict[ind][enc]
                else:
                    if ind not in qr_dict:
                        qr_dict[ind] = {}
                    else:
                        qr_dict[ind][enc] = ind_enc_curv_dict[ind][enc]
        else:
            print('individual %s has only %d encounters' % (
                ind, num_encounters))

    # we only use individuals with at least two encounters
    for qind in qr_dict.keys():
        assert qind in db_dict.keys(), '%s missing from db!' % (qind)
    return db_dict, qr_dict


def separate_sdrp_dataset(fpath_list, ind_list, enc_list, curv_dict):
    # {'i1': {'e1': [v1, v2, ..., vn], 'e2': [v1, v2, ..., vm]}}
    ind_enc_curv_dict = {}
    for fpath, ind, enc in zip(fpath_list, ind_list, enc_list):
        fname = splitext(basename(fpath))[0]
        if fname in curv_dict.keys():
            if ind not in ind_enc_curv_dict:
                ind_enc_curv_dict[ind] = {}
            if enc not in ind_enc_curv_dict[ind]:
                ind_enc_curv_dict[ind][enc] = []
            ind_enc_curv_dict[ind][enc].append(curv_dict[fname])

    db_dict, qr_dict = {}, {}
    individuals = ind_enc_curv_dict.keys()
    num_db_encounters = 5
    for ind in individuals:
        encounters = ind_enc_curv_dict[ind].keys()
        num_encounters = len(encounters)
        if num_encounters > 1:
            if ind not in qr_dict:
                qr_dict[ind] = {}
            if num_encounters <= num_db_encounters:
                num_samples = num_encounters - 1
            else:
                num_samples = num_db_encounters

            rind = np.arange(num_encounters)
            np.random.shuffle(rind)
            db_idx, qr_idx = np.split(rind, np.array([num_samples]), axis=0)
            for idx in db_idx:
                enc = encounters[idx]
                d_curv_list = []
                for curv in ind_enc_curv_dict[ind][enc]:
                    d_curv_list.append(curv)
                db_dict[ind] = d_curv_list
            for idx in qr_idx:
                enc = encounters[idx]
                q_curv_list = []
                for curv in ind_enc_curv_dict[ind][enc]:
                    q_curv_list.append(curv)
                qr_dict[ind][enc] = q_curv_list
        else:
            print('individual %s has only %d encounters' % (
                ind, num_encounters))

    return db_dict, qr_dict
