#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cfloat>
//#include <stdint.h>
#include <ctime>
//#include <cstdint>
#include <Eigen/Dense>
#include <iostream>
//#include <cblas>
// this is bad don't do this kids
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

//#define DLLEXPORT extern "C" __declspec(dllexport)

using namespace Eigen;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> NDArrayFlattened;
typedef Map<NDArrayFlattened> ExternNDArrayf;

typedef Matrix<int, Dynamic, Dynamic, RowMajor> NDArrayFlattenedi;
typedef Map<NDArrayFlattenedi> ExternNDArrayi;

extern "C" void dtw_curvweighted(float * seq1curvv, float * seq2curvv,
                                 int seq1_len, int seq2_len, int window,
                                 int curv_hist_size, float * curv_hist_weightsv,
                                 float * distmat_outv) {

    window = MAX(window, abs(seq1_len - seq2_len) + 1);
    ExternNDArrayf distmat_out((float*)distmat_outv, seq1_len, seq2_len);
    ExternNDArrayf seq1curv((float*)seq1curvv, seq1_len, curv_hist_size);
    ExternNDArrayf seq2curv((float*)seq2curvv, seq2_len, curv_hist_size);


    ExternNDArrayf curv_hist_weights((float*)curv_hist_weightsv, curv_hist_size, 1);

    float dist;
    for (int i = 1; i < seq1_len; i++) {
        for (int j = MAX(1, i - window); j < MIN(seq2_len, i + window); j++) {
            dist = ((seq1curv.row(i).array() * curv_hist_weights.transpose().array()) - 
                    (seq2curv.row(j).array() * curv_hist_weights.transpose().array())).matrix().norm();
            distmat_out(i,j) = dist + MIN(distmat_out(i, j-1), 
                                        MIN(distmat_out(i-1, j),
                                            distmat_out(i-1, j-1)));
        }
    }
}
