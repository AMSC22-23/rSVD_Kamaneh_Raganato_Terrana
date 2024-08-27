#ifndef SVD_H
#define SVD_H

#include <iostream>
#include <random>
#include <limits>
#include <numeric>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
// #include <eigen3/Eigen/Dense>
// #include <eigen3/unsupported/Eigen/MatrixFunctions>
// #include <eigen3/Eigen/Sparse>
// #include <eigen3/unsupported/Eigen/SparseExtra>

using namespace std;
// using namespace Eigen;

using Mat_m = Eigen::MatrixXd;
using Vec_v = Eigen::VectorXd;

class SVD
{
public:
    // Power Method
    void PM(Mat_m &A, Mat_m &B, double &sigma, Vec_v &u, Vec_v &v);

    // Singular Value Decomposition through Power Method
    SVD(Mat_m &A, Vec_v &sigma, Mat_m &U, Mat_m &V, const int dim);
};

#endif