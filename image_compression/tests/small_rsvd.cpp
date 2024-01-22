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

    if (rank == 0) std::cout << "small test rSVD" << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    int k = 2; // numerical rank (we need an algorithm to find it) or target rank
    int p = 0; // oversampling parameter, usually it is set to 5 or 10
    // int l = k + p;
    int n = 100;
    for (int l=2; l<n+1; l+=2){
    

        Eigen::MatrixXd A= Eigen::MatrixXd::Zero(n, n);

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

        // A.setRandom();
        // Diagonal matrix
        // for (int i=0; i<n; i++){
        //     A(i, i) = distribution(gen);
        // }

        // Block Diagonal matrix
        for (int i=0; i<n/4; i++){
            for (int j=0; j<n/4; j++){
                if (i==j) A(i, j) = distribution(gen);
            }
        }

        // for (int i=n/4; i<n/2; i++){
        //     for (int j=n/4; j<n/2; j++){
        //         A(i, j) = distribution(gen);
        //     }
        // }

        double frobeniusNorm = A.norm();
        A /= frobeniusNorm;

        // if (rank == 0) cout << A << endl;

        Mat A_copy = A;

        // if (rank == 0) cout << "A : " << A << endl;

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
        

        // if (rank == 0) cout << "A~: " << A_2 << endl;

        if (rank == 0) {        
            std::cout << "l = " << l << "\n";
            cout << "norm ||A~ - A|| = " << (A_2 - A).norm() << "\n";
            // std::cout << "Execution Time: " << duration.count() << " seconds\n\n";
        }
    }
    MPI_Finalize();
    return 0;
}