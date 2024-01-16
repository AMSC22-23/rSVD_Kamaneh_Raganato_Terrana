#ifndef matrixOperations_H
#define matrixOperations_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

using namespace std;
using namespace Eigen;

using Mat = MatrixXd;
using Vec = VectorXd;

Mat manualMatrixMultiply(const Mat &matrix1, const Mat &matrix2);

#endif