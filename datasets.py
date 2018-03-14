from __future__ import absolute_import, division, print_function
import cv2
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
from os import listdir
from os.path import basename, join, splitext


def load_dataset(name):
    if name == 'nz':
        return load_nz_dataset()
    elif name == 'sdrp':
        return load_sdrp_dataset([2013, 2014, 2015, 2016])
    elif name == 'fb':
        return load_fb_dataset()
    elif name == 'crc':
        return load_crc_dataset()
    elif name == 'fw':
        return load_fw_dataset()
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
                        'Left',  # all images in this dataset are left-views
                    ))

    return data_list


def load_sdrp_dataset(years):
    database = '/home/hendrik/projects/sdrp/databases/sdrp.db'
    conn = sqlite3.connect(database)
    cur = conn.cursor()

    # inner join because we don't want sightings that are not in the catalog
    cur.execute(
        'select '
        '  Image, SurveyNum, Sighting, Date, Alias, Image, '
        '  luImageSide.ImageSide, '
        '  luDistinctiveness.Rating, luDistinctiveness.Distinctiveness, '
        '  luFocus.Code, luFocus.Focus, '
        '  luContrast.Code, luContrast.Contrast, '
        '  luAngle.Code, luAngle.Angle, '
        '  luPartial.Code, luPartial.Partial, '
        '  luDistance.Code, luDistance.Distance '
        'from IndivSighting '
        'inner join Sighting on '
        '  IndivSighting.SightingFID = Sighting.ID '
        'inner join Individuals on '
        '  IndivSighting.IndividualsFID = Individuals.ID '
        'inner join IndivImage on '
        '  IndivImage.IndivSightingFID = IndivSighting.ID '
        'inner join Survey on '
        '  Sighting.SurveyFID = Survey.ID '
        'inner join PhotographicQuality on '
        '  PhotographicQuality.IndivImageFID = IndivImage.ID '
        'inner join luFocus on '
        '  PhotographicQuality.Focus = luFocus.Code '
        'inner join luContrast on '
        '  PhotographicQuality.Contrast = luContrast.Code '
        'inner join luAngle on '
        '  PhotographicQuality.Angle = luAngle.Code '
        'inner join luPartial on '
        '  PhotographicQuality.Partial = luPartial.Code '
        'inner join luDistance on '
        '  PhotographicQuality.Distance = luDistance.Code '
        'inner join luImageSide on '
        '  IndivImage.ImageSide = luImageSide.Code '
        'left outer join luDistinctiveness on '
        '  IndivSighting.Distinctiveness = luDistinctiveness.Code'
    )

    data_dir = '/media/sdrp/SDRP Data/FinBase/Images'
    data_list = []
    for (fname, survey, sighting, date, alias, image, side,
            distinct_rating, distinct_text,
            focus_code, focus_text, contrast_code, contrast_text,
            angle_code, angle_text, partial_code, partial_text,
            distance_code, distance_text) in cur.fetchall():

        date = datetime.strptime(date, '%m/%d/%y %H:%M:%S')
        quality_score = (focus_code + contrast_code + angle_code +
                         partial_code + distance_code)
        #if quality_score > 7:
        #    continue
        if quality_score > 11:  # Q1 or Q2 in quality
            continue
        if distinct_rating not in ['D1', 'D2']:
            continue
        # we only use left-view images for now
        if date.year in years:
            data_list.append((
                join(data_dir, fname),
                alias,
                '%s-%s' % (survey, sighting),
                side,
            ))

    return data_list


def load_fb_dataset():
    data_dir = '/media/hdd/hendrik/datasets/fb-humpbacks'
    csv_fpath = join(data_dir, 'filelist.csv')

    df = pd.read_csv(csv_fpath, header='infer',
                     usecols=['Filepath', 'Individual', 'Encounter', 'Side'])

    filepaths = df['Filepath'].values
    individuals = df['Individual'].values
    encounters = df['Encounter'].values
    sides = df['Side'].values

    data_list = []
    for fpath, indiv, enc, side in zip(
            filepaths, individuals, encounters, sides):
        data_list.append((
            fpath, indiv, enc, side
        ))

    return data_list


def load_fw_dataset():
    root = '/media/hdd/hendrik/datasets/metr/'
    csv_fpath = join(root, 'fw.csv')
    df = pd.read_csv(csv_fpath)
    data_list = []
    # no encounter information, assume all separate
    for enc, (fname, indiv, side) in enumerate(
            df[['Filename', 'ID', 'Side']].values):
        img_fpath = join(root, 'images', fname)
        data_list.append((
            img_fpath,
            indiv,
            enc,
            side,
        ))

    return data_list


def load_crc_dataset():
    data_dir = '/media/hdd/hendrik/datasets/crc-humpbacks'
    dir_names = ['2014a', '2014b', 'reference']

    data_list = []
    for dir_name in dir_names:
        csv_fpath = join(data_dir, 'filelists', '%s.csv' % dir_name)
        df = pd.read_csv(csv_fpath, header='infer',
                         usecols=['Filename', 'CRCID'])
        filenames = df['Filename'].values
        names = df['CRCID'].values
        for enc, (fname, name) in enumerate(zip(filenames, names)):
            fpath = join(data_dir, dir_name, 'images', fname)
            data_list.append((
                fpath, name, enc, 'Left'
            ))

    return data_list


def separate_database_queries(name, fpath_list, ind_list, enc_list, curv_dict,
                              **kwargs):
    if name == 'nz':
        return separate_nz_dataset(fpath_list, ind_list, enc_list, curv_dict)
    # separate the flukebook dataset the same way
    elif name in ('sdrp', 'fb', 'crc'):
        return separate_sdrp_dataset(fpath_list, ind_list, enc_list, curv_dict,
                                     **kwargs)
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
                    qr_dict[ind][enc] = ind_enc_curv_dict[ind][enc]
        else:
            db_dict[ind] = ind_enc_curv_dict[ind][encounters[0]]
            #print('individual %s has only %d encounters' % (
            #    ind, num_encounters))

    # we only use individuals with at least two encounters
    #for qind in qr_dict.keys():
    #    assert qind in db_dict.keys(), '%s missing from db!' % (qind)
    return db_dict, qr_dict


def separate_sdrp_dataset(fpath_list, ind_list, enc_list, curv_dict,
                          num_db_encounters=10):
    # {'i1': {'e1': [v1, v2, ..., vn], 'e2': [v1, v2, ..., vm]}}
    ind_enc_curv_dict = {}
    for fpath, ind, enc in zip(fpath_list, ind_list, enc_list):
        if ind == '0':
            continue
        fname = splitext(basename(fpath))[0]
        # FIXME: this will break if two files only differ in the extension
        if fname in curv_dict.keys():
            if ind not in ind_enc_curv_dict:
                ind_enc_curv_dict[ind] = {}
            if enc not in ind_enc_curv_dict[ind]:
                ind_enc_curv_dict[ind][enc] = []
            ind_enc_curv_dict[ind][enc].append(curv_dict[fname])

    db_dict, qr_dict = {}, {}
    individuals = ind_enc_curv_dict.keys()
    for ind in individuals:
        encounters = ind_enc_curv_dict[ind].keys()
        num_encounters = len(encounters)
        if num_encounters > 1:
            if ind not in qr_dict:
                qr_dict[ind] = {}
            if num_encounters <= num_db_encounters:
                num_db_samples = num_encounters - 1
            else:
                num_db_samples = num_db_encounters

            rind = np.arange(num_encounters)
            np.random.shuffle(rind)
            db_idx, qr_idx = np.split(rind, np.array([num_db_samples]), axis=0)
            d_curv_list = []
            for idx in db_idx:
                enc = encounters[idx]
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
            db_dict[ind] = ind_enc_curv_dict[ind][encounters[0]]
            #single_encounter_individuals += 1

    return db_dict, qr_dict
