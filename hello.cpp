#include <mpi.h>
#include <iostream>
#include <Eigen/Dense>
#include <chrono>

int main() {
    // Define a matrix A of size 3x3
    // Initialize random seed
    srand(time(NULL));

    // Define the size of the matrix
    const int size = 100;

    // Define a dynamic matrix A of size 10x10 and initialize with random values
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(size, size);

    // Define a dynamic vector b of size 10 and initialize with ones
    Eigen::VectorXd b = Eigen::VectorXd::Ones(size);

    // Define a dynamic vector c of size 10
    Eigen::VectorXd c(size);

    auto start_time = std::chrono::high_resolution_clock::now();

    double sum = 0;
    for (int i=0; i<size; i++){
        sum = 0;
        for (int j=0; j<size; j++){
            sum += A(i, j) * b(j);
        }
        c(i) = sum;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Display the matrix A and vector b
    // std::cout << "Matrix A:\n" << A << "\n\n";
    // std::cout << "Vector b:\n" << b << "\n\n";
    // std::cout << "Vector c:\n" << c << "\n\n";
    std::cout << "Execution Time: " << duration.count() << " microseconds\n";

    return 0;
}
