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
    int m = 50;
    int n = 40;
    Mat A = Mat::Random(m, n);
    // A << 1.0, 2.0, 3.0, 4.0,
    //      5.0, 6.0, 7.0, 8.0,
    //      9.0, 10.0, 11.0, 12.0,
    //      13.0, 14.0, 15.0, 16.0;

    int k = 10; // numerical rank (we need an algorithm to find it) or target rank
    int p = 5; // oversampling parameter, usually it is set to 5 or 10
    int l = k + p;
    Mat U = Mat::Zero(A.rows(), l);
    Vet S = Vet::Zero(l);
    Mat V = Mat::Zero(l, A.cols());
    //vet sigma = vet::Zero(dim);
    //mat U = mat::Zero(A.rows(), dim); // U m*n or m*m
    //mat V = mat::Zero(dim, A.cols()); // V n*n or m*n
    rSVD(A, U, S, V);
    std::cout<<"--RSVD TEST--"<<std::endl;
    // std::cout << "A = \n" << A << std::endl;
    // std::cout << "U = \n" << U << std::endl;
    std::cout << "S = \n" << S << std::endl;
    // std::cout << "V = \n" << V << std::endl;
    
    Mat diagonalMatrix = S.asDiagonal();

    // Mat A_2(m, n);
    // A_2 = U * diagonalMatrix;
    // A_2 = A_2 * V;

    // // std::cout << "A_2 = \n" << A_2 << std::endl;

    // Mat diff = A - A_2;
    // double norm_of_difference = (A - A_2).norm();
    // std::cout << "norm of diff : " << norm_of_difference << std::endl;
    return 0;
}