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

    for (int i=0; i<x0.size(); i++) {
        x0(i) = distribution(gen);
    }
    x0.normalize();

    // Define the number of iterations
    double epsilon = 1.e-10;
    double delta = 0.05;
    double lambda = 0.1;
    int s = ceil( log(4*log(2*A.cols()/delta)/(epsilon*delta)) /(2*lambda));

    // Divide rows of A among processors
    int rows_per_proc = B.rows() / num_procs;
    Eigen::MatrixXd local_B = B.block(rank * rows_per_proc, 0, rows_per_proc, B.cols());

    // Compute local portion of the result vector c manually
    Eigen::VectorXd local_res(rows_per_proc);

    for (int outer_index=1; outer_index<=s; outer_index++) {

        // x0 = B*x0; 
        // converting the above multiplication to manual with MPI
        for (int i = 0; i < rows_per_proc; ++i) {
            local_res(i) = 0.0;
            for (int j = 0; j < B.cols(); ++j) {
                local_res(i) += local_B(i, j) * x0(j);
            }
        }

        MPI_Gather(local_res.data(), rows_per_proc, MPI_DOUBLE, res.data(), rows_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            x0 = res;
            x0.normalize();
        }
        MPI_Bcast(x0.data(), x0.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Compute the left singlular vector
    v = x0;
    v.normalize();

    // Compute the singular value
    sigma = (A*v).norm();

    // Compute the right singular vector
    u = A*v/sigma;

}



