#include "matrixOperations.hpp"
#include <Eigen/Dense>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <iostream>

using namespace Eigen;
using namespace std;

using Mat = MatrixXd;
using Vec = VectorXd;

Mat manualMatrixMultiply(const Mat &A, const Mat &B) {
    if (A.cols() != B.rows()) {
        // Matrices dimensions are not suitable for multiplication
        throw std::invalid_argument("Matrices dimensions are not compatible for manual matrix multiplication");
    }

    int resultRows = A.rows();
    int resultCols = B.cols();
    Mat C(resultRows, resultCols);

    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto start = std::chrono::high_resolution_clock::now();

    // Divide rows of A among processors
    int rows_per_proc = A.rows() / num_procs;
    int rows_remainder = A.rows() % num_procs;
    int local_rows = (rank < rows_remainder) ? rows_per_proc + 1 : rows_per_proc;
    int offset = rank * rows_per_proc + std::min(rank, rows_remainder);

    Eigen::MatrixXd local_A = A.block(offset, 0, local_rows, A.cols());
    Eigen::VectorXd c(resultRows);
    
    for (int outer_index=0; outer_index<B.cols(); outer_index++){

        Eigen::VectorXd b(resultRows); 
        b = B.col(outer_index);
        MPI_Bcast(b.data(), b.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        Eigen::VectorXd local_c(local_rows);

        for (int i = 0; i < local_rows; ++i) {
            local_c(i) = 0.0;
            // #pragma omp simd
            for (int j = 0; j < A.cols(); ++j) {
                local_c(i) += local_A(i, j) * b(j);
            }
        }

        // Gather local results using MPI_Gatherv
        // Compute displacements and recvcounts arrays
        std::vector<int> recvcounts(num_procs);
        std::vector<int> displacements(num_procs);

        // #pragma omp simd
        for (int i = 0; i < num_procs; ++i) {
            recvcounts[i] = (i < rows_remainder) ? rows_per_proc + 1 : rows_per_proc;
            displacements[i] = i * rows_per_proc + std::min(i, rows_remainder);
        }

        MPI_Gatherv(local_c.data(), local_rows, MPI_DOUBLE, c.data(), recvcounts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0){
            C.col(outer_index) = c;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    if (rank == 0) {
    // //     // std::cout << "Matrix A:\n" << A << "\n\n";
    // //     // std::cout << "Matrix B:\n" << B << "\n\n";
    // //     // std::cout << "Matrix C (Result):\n" << C << "\n\n";

    // //     // std::cout << "Actual c:\n" << A * B << std::endl;
        std::cout << "norm of diff: " << (C - A*B).norm() << "\n";
        std::cout << "Execution Time: " << duration.count() << " microseconds\n";
    }

    return C;
}

// int main(int argc, char** argv) {
//     // Initialize MPI
//     MPI_Init(&argc, &argv);

//     Eigen::MatrixXd A(300, 300);
//     Eigen::MatrixXd B(300, 300);
//     // Eigen::MatrixXd C(3, 3); 
//     A.setRandom();
//     B.setRandom();
//     // A << 1, 2, 3,
//     //      4, 5, 6,
//     //      10, 8, 9;

//     // B << 1, 1, 1,
//     //      1, 5, 1,
//     //      1, 1, 1;

//     Mat C = manualMatrixMultiply(A, B);

//     MPI_Finalize();
//     return 0;
// }