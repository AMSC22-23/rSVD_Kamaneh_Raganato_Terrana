#ifndef matrixOperations_H
#define matrixOperations_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

/**
 * @brief Perform manual matrix multiplication.
 * @param matrix1 The first matrix operand.
 * @param matrix2 The second matrix operand.
 * @return The result of the matrix multiplication.
 */
Eigen::MatrixXd manualMatrixMultiply(const Eigen::MatrixXd& matrix1, const Eigen::MatrixXd& matrix2);

#endif // matrixOperations_H
