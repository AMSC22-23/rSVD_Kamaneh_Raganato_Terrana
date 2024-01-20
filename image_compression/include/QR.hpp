#ifndef QR_H
#define QR_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>


Eigen::MatrixXd givens_rotation(double a, double b);
void qr_decomposition_full(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R);
void qr_decomposition_reduced(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R);

#endif
