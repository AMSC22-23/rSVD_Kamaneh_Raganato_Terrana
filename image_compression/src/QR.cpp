#include "QR.hpp"
#include "matrixOperations.hpp"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <mpi.h>

using namespace std;
using namespace Eigen;

using Mat = MatrixXd;
using Vec = VectorXd;

template <typename Scalar>
QRDecomposition<Scalar>::QRDecomposition(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A)
    : A_(A) {
    // Constructor implementation if needed
}

template <typename Scalar>
const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& QRDecomposition<Scalar>::getA() const {
    return A_;
}

template <typename Scalar>
void QRDecomposition<Scalar>::givens_rotation(Scalar a, Scalar b, Eigen::Matrix<Scalar, 2, 2>& G) const {

        double r = hypot(a, b);
        double c = a / r;
        double s = -b / r;

        G << c, -s,
            s, c;
}

template <typename Scalar>
QRFullDecomposition<Scalar>::QRFullDecomposition(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A)
    : QRDecomposition<Scalar>(A) {
    // Constructor
}

template <typename Scalar>
void QRFullDecomposition<Scalar>::decompose(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Q,
                                           Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& R) const {
    // Implementation
    int m =  this->getA().rows();
    int n =  this->getA().cols();
    Matrix2d G;

    Q = Mat::Identity(m, m);
    R =  this->getA();

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

template <typename Scalar>
QRReducedDecomposition<Scalar>::QRReducedDecomposition(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A)
    : QRDecomposition<Scalar>(A) {
    // Constructor
}

template <typename Scalar>
void QRReducedDecomposition<Scalar>::decompose(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Q,
                                               Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& R) const {

        int m = this->getA().rows();
        int n = this->getA().cols();
        Matrix2d G;

        Mat Q_temp = Mat::Identity(m, m);
        Mat R_temp = this->getA();

        for (int j = 0; j < std::min(m, n); ++j) {
            // loop over columns
            for (int i = m - 1; i > j; --i) {
                // loop over rows
                if (R_temp(i, j) != 0) {

                    givens_rotation(R_temp(i - 1, j), R_temp(i, j), G);
                    R_temp.block(i - 1, j, 2, n - j) = G * R_temp.block(i - 1, j, 2, n - j);              
                    Q_temp.leftCols(m).middleCols(i - 1, 2) = Q_temp.leftCols(m).middleCols(i - 1, 2) * G.transpose();

                }
            }
        }
        Q = Q_temp.leftCols(n);
        R = R_temp.topRows(n);
}

template <typename Scalar>
QRMPIDecomposition<Scalar>::QRMPIDecomposition(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A)
    : QRDecomposition<Scalar>(A) {
    // Constructor
}

template <typename Scalar>
void QRMPIDecomposition<Scalar>::decompose(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Q,
                                           Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& R) const {
    // Implementation of decompose for MPI-aware decomposition
     int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int m =  this->getA().rows();
    int n =  this->getA().cols();
    Matrix2d G;

    Mat Q_temp = Mat::Identity(m, m);
    Mat R_temp =  this->getA();

    for (int j = 0; j < std::min(m, n); ++j) {
        // loop over columns
        for (int i = m - 1; i > j; --i) {
            // loop over rows
            if (rank==0 && j==0){
                if (R_temp(i, j) != 0) {
                    givens_rotation(R_temp(i - 1, j), R_temp(i, j), G);
                    R_temp.block(i - 1, j, 2, n - j) = G * R_temp.block(i - 1, j, 2, n - j);              
                    Q_temp.leftCols(m).middleCols(i - 1, 2) = Q_temp.leftCols(m).middleCols(i - 1, 2) * G.transpose();

                }
            }
            MPI_Bcast(R_temp.data(), R_temp.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(Q_temp.data(), Q_temp.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            
            if (rank==1 && j==1){
                if (R_temp(i, j) != 0){
                    givens_rotation(R_temp(i - 1, j), R_temp(i, j), G);
                    R_temp.block(i - 1, j, 2, n - j) = G * R_temp.block(i - 1, j, 2, n - j);              
                    Q_temp.leftCols(m).middleCols(i - 1, 2) = Q_temp.leftCols(m).middleCols(i - 1, 2) * G.transpose();
                }                
            }
            MPI_Bcast(R_temp.data(), R_temp.size(), MPI_DOUBLE, 1, MPI_COMM_WORLD);
            MPI_Bcast(Q_temp.data(), Q_temp.size(), MPI_DOUBLE, 1, MPI_COMM_WORLD);
        }
    }
    Q = Q_temp.leftCols(n);
    R = R_temp.topRows(n);
}

// Explicit instantiation of the templates for the supported types (e.g., double, float)
template class QRDecomposition<double>;
template class QRFullDecomposition<double>;
template class QRReducedDecomposition<double>;
template class QRMPIDecomposition<double>;


// void givens_rotation(double a, double b, Matrix2d& G) 
// {
//     double r = hypot(a, b);
//     double c = a / r;
//     double s = -b / r;

//     G << c, -s,
//          s, c;
// }

// void qr_decomposition_full(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R)
// {
//     int m = A.rows();
//     int n = A.cols();
//     Matrix2d G;

//     Q = Mat::Identity(m, m);
//     R = A;

//     for (int j = 0; j < std::min(m, n); ++j) {
//         for (int i = m - 1; i > j; --i) {
//             if (R(i, j) != 0) {
//                 givens_rotation(R(i - 1, j), R(i, j), G);
//                 R.block(i - 1, j, 2, n - j) = G * R.block(i - 1, j, 2, n - j);
//                 Q.leftCols(m).middleCols(i - 1, 2) *= G.transpose();

//             }
//         }
//     }
// }

// void qr_decomposition_reduced(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R)
// {
//     int m = A.rows();
//     int n = A.cols();
//     Matrix2d G;

//     Mat Q_temp = Mat::Identity(m, m);
//     Mat R_temp = A;

//     for (int j = 0; j < std::min(m, n); ++j) {
//         // loop over columns
//         for (int i = m - 1; i > j; --i) {
//             // loop over rows
//             if (R_temp(i, j) != 0) {

//                 givens_rotation(R_temp(i - 1, j), R_temp(i, j), G);
//                 R_temp.block(i - 1, j, 2, n - j) = G * R_temp.block(i - 1, j, 2, n - j);              
//                 Q_temp.leftCols(m).middleCols(i - 1, 2) = Q_temp.leftCols(m).middleCols(i - 1, 2) * G.transpose();

//             }
//         }
//     }
//     Q = Q_temp.leftCols(n);
//     R = R_temp.topRows(n);
// }

// void qr_decomposition_reduced_mpi(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R)
// {
//     int num_procs, rank;
//     MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//     int m = A.rows();
//     int n = A.cols();
//     Matrix2d G;

//     Mat Q_temp = Mat::Identity(m, m);
//     Mat R_temp = A;

//     for (int j = 0; j < std::min(m, n); ++j) {
//         // loop over columns
//         for (int i = m - 1; i > j; --i) {
//             // loop over rows
//             if (rank==0 && j==0){
//                 if (R_temp(i, j) != 0) {
//                     givens_rotation(R_temp(i - 1, j), R_temp(i, j), G);
//                     R_temp.block(i - 1, j, 2, n - j) = G * R_temp.block(i - 1, j, 2, n - j);              
//                     Q_temp.leftCols(m).middleCols(i - 1, 2) = Q_temp.leftCols(m).middleCols(i - 1, 2) * G.transpose();

//                 }
//             }
//             MPI_Bcast(R_temp.data(), R_temp.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
//             MPI_Bcast(Q_temp.data(), Q_temp.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            
//             if (rank==1 && j==1){
//                 if (R_temp(i, j) != 0){
//                     givens_rotation(R_temp(i - 1, j), R_temp(i, j), G);
//                     R_temp.block(i - 1, j, 2, n - j) = G * R_temp.block(i - 1, j, 2, n - j);              
//                     Q_temp.leftCols(m).middleCols(i - 1, 2) = Q_temp.leftCols(m).middleCols(i - 1, 2) * G.transpose();
//                 }                
//             }
//             MPI_Bcast(R_temp.data(), R_temp.size(), MPI_DOUBLE, 1, MPI_COMM_WORLD);
//             MPI_Bcast(Q_temp.data(), Q_temp.size(), MPI_DOUBLE, 1, MPI_COMM_WORLD);
//         }
//     }
//     Q = Q_temp.leftCols(n);
//     R = R_temp.topRows(n);
// }


