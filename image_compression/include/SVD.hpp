#ifndef SVD_H
#define SVD_H

#include "PowerMethod.hpp" 

/**
 * @brief Perform the Singular Value Decomposition (SVD) using MPI for parallelization.
 * @param A The input matrix.
 * @param sigma The singular values.
 * @param U The left singular vectors.
 * @param V The right singular vectors.
 * @param dim The target dimensionality of the decomposition.
 */
void singularValueDecomposition_mpi(Eigen::MatrixXd& A, Eigen::VectorXd& sigma, Eigen::MatrixXd& U, Eigen::MatrixXd& V, const int dim);

/**
 * @brief Perform the Singular Value Decomposition (SVD).
 * @param A The input matrix.
 * @param sigma The singular values.
 * @param U The left singular vectors.
 * @param V The right singular vectors.
 * @param dim The target dimensionality of the decomposition.
 */
void singularValueDecomposition(Eigen::MatrixXd& A, Eigen::VectorXd& sigma, Eigen::MatrixXd& U, Eigen::MatrixXd& V, const int dim);

#endif // SVD_H
