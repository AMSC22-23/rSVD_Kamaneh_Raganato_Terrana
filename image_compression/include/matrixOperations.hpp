#ifndef matrixOperations_H
#define matrixOperations_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

Eigen::MatrixXd manualMatrixMultiply(const Eigen::MatrixXd& matrix1, const Eigen::MatrixXd& matrix2);

#endif