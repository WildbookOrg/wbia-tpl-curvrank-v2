from __future__ import absolute_import, division, print_function
from ibeis.init import sysres
import ibeis


if __name__ == '__main__':
    dbdir = sysres.ensure_testdb_curvrank()
    ibs = ibeis.opendb(dbdir=dbdir)

    db_imageset_rowid = ibs.get_imageset_imgsetids_from_text('database')
    db_aid_list = ibs.get_imageset_aids(db_imageset_rowid)
    qr_imageset_rowid = ibs.get_imageset_imgsetids_from_text('query')
    qr_aid_list = ibs.get_imageset_aids(qr_imageset_rowid)

    score_dict = ibs.ibeis_plugin_curvrank_scores(db_aid_list, qr_aid_list,
                                                  verbose=True)

    score_list = list(score_dict.items())
    score_list.sort(key=lambda score: score[1])

    print('\nResults:')
    for nid, score in score_list:
        name = ibs.get_name_texts(nid)
        print('% 8s: %8.04f' % (name, score, ))
