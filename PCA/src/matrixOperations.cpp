#include "matrixOperations.hpp"
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

Mat manualMatrixMultiply(const Mat &matrix1, const Mat &matrix2) {
    if (matrix1.cols() != matrix2.rows()) {
        // Matrices dimensions are not suitable for multiplication
        throw std::invalid_argument("Matrices dimensions are not compatible for manual matrix multiplication");
    }

    int resultRows = matrix1.rows();
    int resultCols = matrix2.cols();
    Mat result(resultRows, resultCols);

    for (int i = 0; i < resultRows; ++i) {
        for (int j = 0; j < resultCols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < matrix1.cols(); ++k) {
                sum += matrix1(i, k) * matrix2(k, j);
            }
            result(i, j) = sum;
        }
    }

    return result;
}