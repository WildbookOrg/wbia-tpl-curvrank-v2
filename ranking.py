import numpy as np

from pydtw import compute_dtw_fast


def rank_individuals(query_vectors, database, simfunc):
    dbindivs = database.keys()
    scores = np.zeros(len(dbindivs), dtype=np.float32)
    for i, dind in enumerate(dbindivs):
        database_vectors = database[dind]
        s = simfunc(query_vectors, database_vectors)
        scores[i] = s

    asc_scores_idx = np.argsort(scores)
    ranking = [dbindivs[idx] for idx in asc_scores_idx]
    scores = [scores[idx] for idx in asc_scores_idx]
    return ranking, scores


def dtw_alignment_cost(query_vectors, database_vectors, weights, window):
    S = np.zeros((len(query_vectors), len(database_vectors)), dtype=np.float32)
    for i, qcurv in enumerate(query_vectors):
        for j, dcurv in enumerate(database_vectors):
            S[i, j] = compute_dtw_fast(qcurv, dcurv, weights, window)

    return S.min(axis=None)
