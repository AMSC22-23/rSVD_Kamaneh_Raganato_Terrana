#include "SVD.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <mpi.h>
#include <iostream>
#include <filesystem> // C++17 or later
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>

using namespace std;

// The main function for running the tests

int main(int argc, char** argv) {
    std::cout << "small test SVD" << std::endl;
    MPI_Init(&argc, &argv);

    Eigen::MatrixXd A(4, 4);
    A << 1, 2, 3, 4,
         1, 5, 6, 9,
         7, 8, 9, 5,
         1, 8, 3, 7;

    Mat A_copy = A;
    int m = A.rows();
    int n = A.cols();
    Mat U = Mat::Zero(m, m);
    Vec S = Vec::Zero(m);
    Mat V = Mat::Zero(m, n);
    int min= m < n ? m : n;

    SVD(A, S, U, V, min);
    // cout << "U: " << U << endl;
    // cout << "S: " << S << endl;
    // cout << "V: " << V << endl;

    Mat diagonalMatrix = S.asDiagonal();
    Mat A_2(m, n);
    Mat mid(m, m);
    mid = U * diagonalMatrix;
    A_2 = mid * V.transpose();

    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank==0) cout << "A_2: \n" << A_2 << endl;

    Mat diff = A_copy - A_2;
    // cout << "A: " << A_copy << endl;
    // cout << diff << endl;
    double norm_of_difference = (diff).norm();

    if (rank ==0) cout << "norm : " << norm_of_difference << endl;
    MPI_Finalize();
    return 0;
}