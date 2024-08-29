#include "rSVD.hpp"
#include <iostream>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <mpi.h>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <filesystem> // C++17 or later
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>


// The main function for running the tests

int main(int argc, char** argv) {
    using namespace std;
    using namespace Eigen;

    using Mat = MatrixXd;
    using Vec = VectorXd;

    MPI_Init(&argc, &argv);
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank==0) std::cout << "*** rSVD test 2 ***\n" << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    int k = 2; // numerical rank (we need an algorithm to find it) or target rank
    int p = 1; // oversampling parameter, usually it is set to 5 or 10
    int l = k + p;
    int n = 5;
    
    Eigen::MatrixXd A= Eigen::MatrixXd::Zero(n, n);

    A.setRandom();
    Mat A_copy = A;

    if (rank == 0) cout << "A : " << A << "\n\n";

    Mat U = Mat::Zero(n, l);
    Vec S = Vec::Zero(l);
    Mat V = Mat::Zero(l, n);

    auto start = std::chrono::high_resolution_clock::now();
    rSVD(A, U, S, V, l);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // cout << "U: \n" << U << endl;
    // cout << "S: \n" << S << endl;
    // cout << "V: \n" << V << endl;

    Mat diagonalMatrix = S.asDiagonal();
    Mat mid(n, l);
    mid = U * diagonalMatrix;
    Mat A_2 = mid * V.transpose();
        

    if (rank == 0) cout << "A~: " << A_2 << "\n\n";

    if (rank==0) {
        cout << "norm ||A~ - A|| = " << (A_2 - A).norm() << "\n";
        cout << "Execution Time: " << duration.count() << " seconds" << endl;
    }
    MPI_Finalize();
    return 0;
}