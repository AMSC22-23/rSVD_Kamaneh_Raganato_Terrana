#include <iostream>
#include <mpi.h>
#include <Eigen/Dense>
#include "rSVD.hpp"

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) std::cout << "main running" << std::endl;

    // int m = A.rows();
    // int n = A.cols();
    // int k = 0; // numerical rank (we need an algorithm to find it) or target rank
    // int p = 16; // oversampling parameter, usually it is set to 5 or 10
    // int l = k + p;
    // Mat A_copy = A;
    // Mat U = Mat::Zero(m, l);
    // Vec S = Vec::Zero(l);
    // Mat V = Mat::Zero(l, n);
    // rSVD(A, U, S, V, l);


    MPI_Finalize();
    return 0;
}
