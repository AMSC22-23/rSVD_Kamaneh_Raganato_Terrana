#include "SVD_class.hpp"
#include <chrono>


int main() {
    // Create a random 100x100 matrix
    MPI_Init(NULL, NULL);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    Mat A = Mat::Random(216, 216);

    // Create an SVD object
    bool full_matrices = false;
    SVD svd(A, full_matrices);

    // Measure the time taken by compute()
    /*auto start = std::chrono::high_resolution_clock::now();
    svd.compute();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time taken by compute(): " << diff.count() << " s\n";*/

    // Measure the time taken by rSVD()
    int target_rank = 50;
     auto start = std::chrono::high_resolution_clock::now();
    svd.rSVD(target_rank,rank,num_procs);
     auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;

    double max_time,diff_count=diff.count();
    MPI_Reduce(&diff_count, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Time taken by rSVD(): " << max_time << " s\n";
    }
    //std::cout << "Time taken by rSVD(): " << diff.count() << " s\n";

    // Measure the time taken by JacobiSVD()
    /*start = std::chrono::high_resolution_clock::now();
    svd.JacobiSVD(A);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Time taken by JacobiSVD(): " << diff.count()<< " s\n";*/

    MPI_Finalize();


    return 0;
}