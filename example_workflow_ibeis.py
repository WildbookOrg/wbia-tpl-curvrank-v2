# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia.init import sysres
import wbia


if __name__ == '__main__':
    dbdir = sysres.ensure_testdb_curvrank()
    ibs = wbia.opendb(dbdir=dbdir)

    db_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Dorsal Database')
    db_aid_list = ibs.get_imageset_aids(db_imageset_rowid)
    qr_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Dorsal Query')
    qr_aid_list = ibs.get_imageset_aids(qr_imageset_rowid)

    score_dict = ibs.wbia_plugin_curvrank_scores(db_aid_list, qr_aid_list,
                                                  verbose=True)

    score_list = list(score_dict.items())
    score_list.sort(key=lambda score: score[1])

    print('\nResults:')
    for nid, score in score_list:
        name = ibs.get_name_texts(nid)
        print('% 8s: %8.04f' % (name, score, ))
