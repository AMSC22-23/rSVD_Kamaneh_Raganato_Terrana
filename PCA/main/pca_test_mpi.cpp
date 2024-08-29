#include <iostream>
#include <chrono>
#include "class_pca_mpi.hpp"
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Create a PCA object
    Mat data = PCA::loadDataset("dataset_athletic.txt");
    bool use_svd = true;
    bool full_matrices = true;

    PCA pca(data,use_svd,full_matrices,rank, size);

    // Load the dataset
    

    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

    // Compute PCA
   // Mat cov = pca.computeCovarianceMatrix();
    pca.compute();
    

    // Stop the timer
    auto stop = std::chrono::high_resolution_clock::now();


    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // Print the execution time
    if (rank == 0) {
        std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
        std::cout << "scores " << pca.scores()<< std::endl;
    }

    MPI_Finalize();
    return 0;
}