#ifndef POWER_METHOD_H
#define POWER_METHOD_H

#include <iostream>
#include <fstream>
#include <random>
#include <mpi.h>
#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

/**
 * @brief Perform the Power Method using MPI for parallelization.
 * @param A The matrix A in the eigenvalue problem A*x = B*x*sigma.
 * @param B The matrix B in the eigenvalue problem A*x = B*x*sigma.
 * @param sigma The estimated eigenvalue.
 * @param u The estimated left eigenvector.
 * @param v The estimated right eigenvector.
 */
void powerMethod_mpi(Eigen::MatrixXd& A, Eigen::MatrixXd& B, double& sigma, Eigen::VectorXd& u, Eigen::VectorXd& v);

/**
 * @brief Perform the Power Method without MPI.
 * @param A The matrix A in the eigenvalue problem A*x = B*x*sigma.
 * @param B The matrix B in the eigenvalue problem A*x = B*x*sigma.
 * @param sigma The estimated eigenvalue.
 * @param u The estimated left eigenvector.
 * @param v The estimated right eigenvector.
 */
void powerMethod(Eigen::MatrixXd& A, Eigen::MatrixXd& B, double& sigma, Eigen::VectorXd& u, Eigen::VectorXd& v);

#endif // POWER_METHOD_H
