#ifndef QR_H
#define QR_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

/**
 * @brief Compute the Givens rotation matrix.
 * @param a The first element of the vector to be rotated.
 * @param b The second element of the vector to be rotated.
 * @return The Givens rotation matrix.
 */
Eigen::MatrixXd givens_rotation(double a, double b);

/**
 * @brief Perform full QR decomposition of a matrix.
 * @param A The matrix to be decomposed.
 * @param Q The orthogonal matrix in the QR decomposition.
 * @param R The upper triangular matrix in the QR decomposition.
 */
void qr_decomposition_full(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R);

/**
 * @brief Perform reduced QR decomposition of a matrix.
 * @param A The matrix to be decomposed.
 * @param Q The orthogonal matrix in the QR decomposition.
 * @param R The upper triangular matrix in the QR decomposition.
 */
void qr_decomposition_reduced(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R);

void qr_decomposition_reduced_mpi(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R); 

#endif // QR_H
