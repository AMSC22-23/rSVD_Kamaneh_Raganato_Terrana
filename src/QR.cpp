#include "QR.hpp"
#include "matrixOperations.hpp"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

using namespace Eigen;
using namespace std;

void givens_rotation(double a, double b, Matrix2d& G) 
{
    double r = hypot(a, b);
    double c = a / r;
    double s = -b / r;

    G << c, -s,
         s, c;
}

void qr_decomposition_full(const Mat &A, Mat &Q, Mat &R) 
{
    int m = A.rows();
    int n = A.cols();
    Matrix2d G;

    Q = Mat::Identity(m, m);
    R = A;

    for (int j = 0; j < std::min(m, n); ++j) {
        for (int i = m - 1; i > j; --i) {
            if (R(i, j) != 0) {
                givens_rotation(R(i - 1, j), R(i, j), G);
                R.block(i - 1, j, 2, n - j) = G * R.block(i - 1, j, 2, n - j);
                Q.leftCols(m).middleCols(i - 1, 2) *= G.transpose();

            }
        }
    }
}

void qr_decomposition_reduced(const Mat &A, Mat &Q, Mat &R) 
{
    int m = A.rows();
    int n = A.cols();
    Matrix2d G;

    Mat Q_temp = Mat::Identity(m, m);
    Mat R_temp = A;

    for (int j = 0; j < std::min(m, n); ++j) {
        for (int i = m - 1; i > j; --i) {
            if (R_temp(i, j) != 0) {
                givens_rotation(R_temp(i - 1, j), R_temp(i, j), G);
                
                // R_temp.block(i - 1, j, 2, n - j) = manualMatrixMultiply(G, R_temp.block(i - 1, j, 2, n - j));
                // this matrix block has 2 rows and n-j cols
                int block_rows = 2;
                int block_cols = n-j;
                Mat temp_1(block_rows, block_cols);
                for (int inner_i=0; inner_i<block_rows; ++inner_i){
                    for (int inner_j=0; inner_j<block_cols; ++inner_j){
                        double sum = 0.0;
                        for (int k = 0; k<block_rows; ++k){
                            sum += G(inner_i, k) * R_temp.block(i - 1, j, 2, n - j)(k, inner_j);
                        }
                        temp_1(inner_i, inner_j) = sum;
                    }
                }

                R_temp.block(i - 1, j, 2, n - j) = temp_1;
                Q_temp.leftCols(m).middleCols(i - 1, 2) = manualMatrixMultiply(Q_temp.leftCols(m).middleCols(i - 1, 2), G.transpose());

            }
        }
    }
    Q = Q_temp.leftCols(n);
    R = R_temp.topRows(n);
}

