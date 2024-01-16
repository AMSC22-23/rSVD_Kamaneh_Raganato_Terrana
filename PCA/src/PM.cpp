#include "PM.hpp"
#include <mpi.h>

void PM(Mat &A, Mat &B, double &sigma, Vec &u, Vec &v) {

    // Get the total number of processors and the rank of the current processor
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Generate a random initial guess x0
    Vec x0 = Vec::Zero(A.cols()); // To multiply B with x0
    Vec res = Vec::Zero(A.cols()); // To multiply B with x0

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < x0.size(); i++) {
        x0(i) = distribution(gen);
    }
    x0.normalize();

    // Define the number of iterations
    double epsilon = 1.e-10;
    double delta = 0.05;
    double lambda = 0.1;
    int s = ceil(log(4 * log(2 * A.cols() / delta) / (epsilon * delta)) / (2 * lambda));

    // Divide rows of A among processors
    int rows_per_proc = B.rows() / num_procs;
    int remaining_rows = B.rows() % num_procs;
    int start_row = rank * rows_per_proc + min(rank, remaining_rows);
    int end_row = start_row + rows_per_proc + (rank < remaining_rows ? 1 : 0);
    Eigen::MatrixXd local_B = B.block(start_row, 0, end_row - start_row, B.cols());

    // Compute local portion of the result vector c manually
    Eigen::VectorXd local_res(end_row - start_row);

    for (int outer_index = 1; outer_index <= s; outer_index++) {

        // x0 = B*x0; 
        // converting the above multiplication to manual with MPI
        for (int i = 0; i < local_res.size(); ++i) {
            local_res(i) = 0.0;
            for (int j = 0; j < B.cols(); ++j) {
                local_res(i) += local_B(i, j) * x0(j);
            }
        }

        // Gather the local results using MPI_Gatherv
        std::vector<int> recvcounts(num_procs);
        std::vector<int> displs(num_procs);

        for (int i = 0; i < num_procs; ++i) {
            recvcounts[i] = rows_per_proc + (i < remaining_rows ? 1 : 0);
            displs[i] = i * rows_per_proc + min(i, remaining_rows);
        }

        MPI_Gatherv(local_res.data(), local_res.size(), MPI_DOUBLE, res.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            x0 = res;
            x0.normalize();
        }

        // Broadcast the updated x0 to all processors
        MPI_Bcast(x0.data(), x0.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Compute the left singular vector
    v = x0;
    v.normalize();

    // Compute the singular value
    sigma = (A * v).norm();

    // Compute the right singular vector
    u = A * v / sigma;
}
