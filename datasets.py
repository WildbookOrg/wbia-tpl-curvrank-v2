import cv2
import sqlite3
from os import listdir
from os.path import join

import os


def load_dataset(name):
    if name == 'nz':
        return load_nz_dataset()
    elif name == 'sdrp':
        return load_sdrp_dataset()
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


def load_sdrp_dataset():
    image_dir = '/media/sdrp/SDRP Data/FinBase/Images'
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

    # list of tuples: (SurveyNum, Sighting, Date, Alias, Image, ImageSide)
    result = cur.fetchall()
    data_list = []
    for (SurveyNum, Sighting, Date, Alias, Image, ImageSide) in result:
        data_list.append((
            join(image_dir, Image),
            Alias,
            '%s-%s' % (SurveyNum, Sighting)
        ))

    conn.commit()
    cur.close()

    return data_list
