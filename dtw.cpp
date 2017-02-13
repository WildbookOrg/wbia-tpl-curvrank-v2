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
  MapType W((float*) w, m, n);

  ArrayType W1 = (X1 - W).abs();
  ArrayType W2 = (X2 - W).abs();
  ArrayType xi, wi;
  ArrayType xj, wj;

  float cost;
  for (int i = 1; i < m; ++i) {
    for (int j = std::max(1, i - window); j < std::min(m, i + window); ++j) {
      xi = X1.row(i);
      xj = X2.row(j);
      wi = W1.row(i);
      wj = W2.row(j);

      cost = ((xi - xj) * (xi - xj) / (wi + wj + 1e-6)).sum();
      costs(i, j) = cost + std::min(costs(i, j - 1),
          std::min(costs(i - 1, j), costs(i - 1, j - 1)));
    }
  }
} 

extern "C" void euclidean(float* x1, float* x2, float* w,
                          int m, int n, int window,
                          float* costs_out) {
  MapType costs((float*) costs_out, m, m);

  MapType X1((float*) x1, m, n);
  MapType X2((float*) x2, m, n);
  MapType weights((float*) w, 1, n);

  ArrayType xi, xj;

  float cost;
  for (int i = 1; i < m; ++i) {
    for (int j = std::max(1, i - window); j < std::min(m, i + window); ++j) {
      xi = X1.row(i);
      xj = X2.row(j);

      cost = sqrt((weights * (xi - xj) * (xi - xj)).sum());
      costs(i, j) = cost + std::min(costs(i, j - 1),
          std::min(costs(i - 1, j), costs(i - 1, j - 1)));
    }
  }
} 
