#ifndef QR_H
#define QR_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

using namespace std;
using namespace Eigen;

using Mat = MatrixXd;
using Vec = VectorXd;

Mat givens_rotation(double a, double b);
void qr_decomposition_full(const Mat &A, Mat &Q, Mat &R);
void qr_decomposition_reduced(const Mat &A, Mat &Q, Mat &R);

#endif
