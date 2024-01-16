#include <iostream>
#include <Eigen/Dense>
#include <mpi.h>
#include <chrono>

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the total number of processors and the rank of the current processor
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialize random seed
    srand(time(NULL));

    // Define the size of the matrices
    const int rows = 50;
    const int cols = 50;

    // Define dynamic matrices A, B, and C of size rows x cols and initialize with values
    Eigen::MatrixXd A(rows, cols);
    Eigen::MatrixXd B(rows, cols);
    Eigen::MatrixXd C(rows, cols);        // Result matrix

    A.setRandom();
    B.setRandom();
    // A << 1, 2, 3,
    //      4, 5, 6,
    //      10, 8, 9;

    // B << 1, 1, 1,
    //      1, 5, 1,
    //      1, 1, 1;

    // Divide rows of A among processors
    int rows_per_proc = A.rows() / num_procs;
    int rows_remainder = A.rows() % num_procs;
    int local_rows = (rank < rows_remainder) ? rows_per_proc + 1 : rows_per_proc;
    int offset = rank * rows_per_proc + std::min(rank, rows_remainder);

    Eigen::MatrixXd local_A = A.block(offset, 0, local_rows, A.cols());
    // Broadcast vector b to all processors
    Eigen::VectorXd c(rows);

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int outer_index=0; outer_index<B.cols(); outer_index++){

        Eigen::VectorXd b(rows); 
        b = B.col(outer_index);
        MPI_Bcast(b.data(), b.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        Eigen::VectorXd local_c(local_rows);

        for (int i = 0; i < local_rows; ++i) {
            local_c(i) = 0.0;
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

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    // Display the matrices A, B, and C
    if (rank == 0) {
        // std::cout << "Matrix A:\n" << A << "\n\n";
        // std::cout << "Matrix B:\n" << B << "\n\n";
        // std::cout << "Matrix C (Result):\n" << C << "\n\n";

        // std::cout << "Actual c:\n" << A * B << std::endl;
        std::cout << "norm of diff: " << (C - A*B).norm() << "\n\n";
        std::cout << "Execution Time: " << duration.count() << " microseconds\n";
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
