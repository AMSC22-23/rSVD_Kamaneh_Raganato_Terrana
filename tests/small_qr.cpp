#include "QR.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <iostream>
#include <filesystem> // C++17 or later
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>

using namespace std;

// The main function for running the tests

int main(int /*argc*/, char** /*argv*/) {
    std::cout << "small test QR" << std::endl;

    Eigen::MatrixXd A(4, 3);
    // A << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    //     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    //     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    //     31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    //     41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    //     51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    //     61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
    //     71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    //     81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
    //     91, 92, 93, 94, 95, 96, 97, 98, 99, 100;

    // A << 0.8147, 0.0975, 0.1576,
    //     0.9058, 0.2785, 0.9706,
    //     0.1270, 0.5469, 0.9572,
    //     0.9134, 0.9575, 0.4854,
    //     0.6324, 0.9649, 0.8003;


    A << 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0;

    int m = A.rows();
    int n = A.cols();
    Mat Q = Mat::Identity(m, std::min(m, n));
    Mat R = A.topRows(std::min(m, n));
    qr_decomposition_reduced(A, Q, R);
    
    cout << "Q: \n" << Q << endl;
    cout << "R: \n" << R << endl;

    Mat A_tilde = Q * R;
    cout << "A~ :\n" << A_tilde << endl;


    return 0;
}