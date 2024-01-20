#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <mpi.h>


int main() {
    MPI_Init(NULL, NULL);
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    const int n = 10; 
    const int k = 5;  
    Eigen::MatrixXd Omega=Eigen::MatrixXd::Constant(n, k,0.0); 
    
    int rows_per_proc = Omega.rows() / num_procs;
    int rows_remainder = Omega.rows() % num_procs;
    int local_rows = (rank < rows_remainder) ? rows_per_proc + 1 : rows_per_proc;
    int offset = rank * rows_per_proc + std::min(rank, rows_remainder);

    // Initialization of the random number generator on each process
    std::random_device rd;
    //use the rank of the process as the seed
    std::mt19937 gen(rd() + rank); 
    std::normal_distribution<double> dist(0.0, 1.0);

    // Generate local portion of the matrix Omega
    Eigen::MatrixXd local_Omega = Omega.block(offset, 0, local_rows, Omega.cols());
    for (int i = 0; i < local_Omega.rows(); ++i) {
        for (int j = 0; j < k; ++j) {
            local_Omega(i, j) = dist(gen);
        }
    }
    
    std::vector<int> recvcounts(num_procs);
    std::vector<int> displacements(num_procs);

    // Compute displacements and recvcounts arrays
    for (int i = 0; i < num_procs; ++i) {
        recvcounts[i] = (i < rows_remainder) ? (rows_per_proc + 1)*k : rows_per_proc*k;
        displacements[i] = (i < rows_remainder) ? (k * (i * (rows_per_proc + 1))) : (k * (rows_remainder * (rows_per_proc + 1)) + (i - rows_remainder) * k * rows_per_proc);    
    }
    
    // Gather local results using MPI_Gatherv
    MPI_Gatherv(local_Omega.data(), k*local_rows, MPI_DOUBLE, Omega.data(), recvcounts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Print the result on the root process
    if (rank == 0) {
        
        std::cout << Omega << std::endl;
        
    }
    
    MPI_Finalize();
   
    return 0;
}