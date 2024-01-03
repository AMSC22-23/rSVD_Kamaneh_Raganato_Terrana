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

    // Define the size of the matrix
    const int size = 100;

    // Define a dynamic matrix A of size 10x10 and initialize with random values
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(size, size);
    // A << 1, 2, 3, 4,
    //     2, 2, 2, 3,
    //     1, 5, 5, 1,
    //     8, 8, 1, 0;

    // Define a dynamic vector b of size 10 and initialize with ones
    Eigen::VectorXd b = Eigen::VectorXd::Ones(size);

    Eigen::VectorXd c(size);

    // Divide rows of A among processors
    int rows_per_proc = A.rows() / num_procs;
    Eigen::MatrixXd local_A = A.block(rank * rows_per_proc, 0, rows_per_proc, A.cols());

    // Broadcast vector b to all processors
    MPI_Bcast(b.data(), b.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute local portion of the result vector c manually
    Eigen::VectorXd local_c(rows_per_proc);

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < rows_per_proc; ++i) {
        local_c(i) = 0.0;
        for (int j = 0; j < A.cols(); ++j) {
            local_c(i) += local_A(i, j) * b(j);
        }
    }
    
    MPI_Gather(local_c.data(), rows_per_proc, MPI_DOUBLE, c.data(), rows_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Display the matrix A, vector b, and vector c
    if (rank == 0) {
        // std::cout << "Matrix A:\n" << A << "\n\n";
        // std::cout << "Vector b:\n" << b << "\n\n";
        // std::cout << "Vector c:\n" << c << "\n\n";
        std::cout << "Execution Time: " << duration.count() << " microseconds\n";
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
