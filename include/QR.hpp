#ifndef QR_H
#define QR_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include "matrixOperations.hpp"
using namespace std;
using namespace Eigen;

using Mat_m = MatrixXd;
using Vec_v = VectorXd;

Mat_m givens_rotation(double a, double b);
void qr_decomposition_full(const Mat_m &A, Mat_m &Q, Mat_m &R);
void qr_decomposition_reduced(const Mat_m &A, Mat_m &Q, Mat_m &R);

#endif
