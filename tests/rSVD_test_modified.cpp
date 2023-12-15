#include "../include/r_SVD/rSVD.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <iostream>
#include <filesystem> // C++17 or later
#include <Eigen/Dense>

// The main function for running the tests

int main(){
    std::cout << "test rSVD" << std::endl;
    int m = 5;
    int n = 10;
    mat A = mat::Random(m, n);
    // A << 1.0, 2.0, 3.0, 4.0,
    //      5.0, 6.0, 7.0, 8.0,
    //      9.0, 10.0, 11.0, 12.0,
    //      13.0, 14.0, 15.0, 16.0;
    // mat U = mat::Zero(m, n);
    // vet S = vet::Zero(n);
    // mat V = mat::Zero(n, n);

    int dim = (A.rows() < A.cols()) ? A.rows() : A.cols();
    vet S = vet::Zero(dim);
    mat U = mat::Zero(A.rows(), dim); // U m*n
    mat V = mat::Zero(dim, A.cols()); // V n*n

    rSVD(A, U, S, V);
    std::cout << "A = \n" << A << std::endl;
    std::cout << "U = \n" << U << std::endl;
    std::cout << "S = \n" << S << std::endl;
    std::cout << "V = \n" << V << std::endl;

    mat diagonalMatrix = mat::Zero(n, n);
    diagonalMatrix = S.asDiagonal(); // n*n
    std::cout << "diagonalMatrix = \n" << diagonalMatrix << std::endl;

    mat A_2(m, n);
    A_2 = U * diagonalMatrix; // A_2 = (m*n) * (n*n) = m*n
    A_2 = A_2 * V.transpose(); // A_2 = (m*n) * (n*n) = m*n
    std::cout << "A_2 = \n" << A_2 << std::endl;

    // mat diff = A - A_2;
    double norm_of_difference = (A - A_2).norm();
    std::cout << "norm of diff : " << norm_of_difference << std::endl;

    return 0;
}