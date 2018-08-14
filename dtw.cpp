#include <iostream>
#include <algorithm>
#include <Eigen/Dense>
#include <math.h>

typedef Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ArrayType;
typedef Eigen::Map<ArrayType> MapType;

extern "C" void weighted_chi_square(float* x1, float* x2, float* w,
                                    int m, int n, int window,
                                    float* costs_out) {
  MapType costs((float*) costs_out, m, m);

  MapType X1((float*) x1, m, n);
  MapType X2((float*) x2, m, n);
  MapType weights((float*) w, m, 1);

  // pre-computed offsets
  ArrayType X1p = (X1 - 0.5).abs();
  ArrayType X2p = (X2 - 0.5).abs();

  ArrayType xi, xj, xip, xjp;
  float wi, wj;

  float cost;
  for (int i = 1; i < m; ++i) {
    for (int j = std::max(1, i - window); j < std::min(m, i + window); ++j) {
      // original points
      xi = X1.row(i);
      xj = X2.row(j);
      // pre-computed offsets with abs. value
      xip = X1p.row(i);
      xjp = X2p.row(j);
      // spatial weights
      wi = weights(i, 0);
      wj = weights(j, 0);

      cost = wi * wj * ((xi - xj) * (xi - xj) / (xip + xjp + 1e-6)).sum();
      costs(i, j) = cost + std::min(costs(i, j - 1),
          std::min(costs(i - 1, j), costs(i - 1, j - 1)));
    }
  }
}

extern "C" void weighted_euclidean(float* x1, float* x2, float* w,
                                   int m, int n, int window,
                                   float* costs_out) {
  MapType costs((float*) costs_out, m, m);

  MapType X1((float*) x1, m, n);
  MapType X2((float*) x2, m, n);
  MapType weights((float*) w, m, 1);

  ArrayType xi, xj;
  float wi, wj;

  float cost;
  for (int i = 1; i < m; ++i) {
    for (int j = std::max(1, i - window); j < std::min(m, i + window); ++j) {
      xi = X1.row(i);
      xj = X2.row(j);

      wi = weights(i, 0);
      wj = weights(j, 0);

      cost = wi * wj * sqrt(((xi - xj) * (xi - xj)).sum());
      costs(i, j) = cost + std::min(costs(i, j - 1),
          std::min(costs(i - 1, j), costs(i - 1, j - 1)));
    }
  }
}
