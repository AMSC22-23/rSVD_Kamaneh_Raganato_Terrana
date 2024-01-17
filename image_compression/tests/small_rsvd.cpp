#include "rSVD.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <mpi.h>
#include <iomanip>
#include <iostream>
#include <filesystem> // C++17 or later
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>

using namespace std;

// The main function for running the tests

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) std::cout << "small test rSVD" << std::endl;
    int m = 6;
    int n = 6;

    Eigen::MatrixXd A(m, n);
    Eigen::MatrixXd X(m, n);
    Eigen::MatrixXd D(m, n);


    // A << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    //     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    //     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    //     31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    //     41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    //     51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    //     61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
    //     71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    //     81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
    //     91, 92, 93, 94, 95, 96, 97, 98, 99, 100;

    // A << 1, 2, 3,
    //     4, 5, 6,
    //     7, 8, 9;

    A.setRandom();

    for (int i=0; i<m; i++){
        for (int j=0; j<n; j++){
            D(i, j) = (j*n + i) * 0.01;
        }
    }

    X = A - D;

    if (rank == 0) cout << A << endl;

    Mat A_copy = A;
    Mat D_copy = D;

    int k = 2; // numerical rank (we need an algorithm to find it) or target rank
    int p = 0; // oversampling parameter, usually it is set to 5 or 10
    int l = k + p;

    // if (rank == 0) cout << "A : " << A << endl;

    Mat U = Mat::Zero(m, l);
    Vec S = Vec::Zero(l);
    Mat V = Mat::Zero(l, n);

    rSVD(D, U, S, V, l);
    // cout << "U: \n" << U << endl;
    // cout << "S: \n" << S << endl;
    // cout << "V: \n" << V << endl;

    Mat diagonalMatrix = S.asDiagonal();
    Mat D_2(m, n);
    Mat mid(m, l);
    mid = U * diagonalMatrix;
    D_2 = mid * V.transpose();
    Mat A_2 = X + D_2;

    if (rank == 0) cout << "A~: " << A_2 << endl;
    Mat diff = D_copy - D_2;
    double norm_of_difference = (diff).norm();

    if (rank == 0) {
        cout << "norm ||D~ - D|| = " << norm_of_difference << endl;
        cout << "norm ||A~ - A|| = " << (A_2 - A).norm() << endl;
    }
    MPI_Finalize();
    return 0;
}