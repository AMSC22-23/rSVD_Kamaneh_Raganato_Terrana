#ifndef rSVD_H
#define rSVD_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

/**
 * @brief Perform an intermediate step for the randomized Singular Value Decomposition (rSVD) using MPI.
 * @param A The input matrix.
 * @param Q The orthogonal matrix in the decomposition.
 * @param Omega The matrix representing random projections.
 * @param l The rank parameter for the decomposition.
 * @param q The number of random projections.
 */
void intermediate_step_mpi(Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& Omega, int &l, int &q);

/**
 * @brief Perform the randomized Singular Value Decomposition (rSVD) using MPI.
 * @param A The input matrix.
 * @param U The left singular vectors.
 * @param S The singular values.
 * @param V The right singular vectors.
 * @param l The rank parameter for the decomposition.
 */
void rSVD_mpi(Eigen::MatrixXd& A, Eigen::MatrixXd& U, Eigen::VectorXd& S, Eigen::MatrixXd& V, int l);

/**
 * @brief Perform an intermediate step for the randomized Singular Value Decomposition (rSVD).
 * @param A The input matrix.
 * @param Q The orthogonal matrix in the decomposition.
 * @param Omega The matrix representing random projections.
 * @param l The rank parameter for the decomposition.
 * @param q The number of random projections.
 */
void intermediate_step(Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& Omega, int &l, int &q);

/**
 * @brief Perform the randomized Singular Value Decomposition (rSVD).
 * @param A The input matrix.
 * @param U The left singular vectors.
 * @param S The singular values.
 * @param V The right singular vectors.
 * @param l The rank parameter for the decomposition.
 */
void rSVD(Eigen::MatrixXd& A, Eigen::MatrixXd& U, Eigen::VectorXd& S, Eigen::MatrixXd& V, int l);

#endif // rSVD_H
