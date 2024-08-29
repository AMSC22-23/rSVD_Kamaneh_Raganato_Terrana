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

    return C;
}
