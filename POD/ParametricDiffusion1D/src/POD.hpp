#ifndef POD_H
#define POD_H

#include <iostream>
#include <random>
#include <limits>
#include <numeric>
#include <tuple>
#include <memory>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

#include "../../../include/SVD_class.hpp"
#include "../../../include/rSVD.hpp"

using namespace std;

using Mat_m = Eigen::MatrixXd;
using Vec_v = Eigen::VectorXd;

class POD
{
public:
    // Default constructor for POD
    POD();

    // Constructor for naive POD
    POD(Mat_m &S, const int r, const int svd_type);

    // Constructor for standard POD
    POD(Mat_m &S, const int r, const double tol, const int svd_type);

    // Constructor for energy POD
    POD(Mat_m &S, Mat_m &Xh, const int r, const double tol, const int svd_type);

    // Constructor for weight POD
    POD(Mat_m &S, Mat_m &Xh, Mat_m &D, const int r, const double tol, const int svd_type);

    // Constructor for online POD through incremental SVD: starting from A it computes U, Sigma, V
    POD(Mat_m &A, Mat_m &U, Mat_m &Sigma, Mat_m &V, const int dim, Vec_v c, const int M, const int r, const double tol, const double tol_sv);

    // Matrix containing the POD modes
    Mat_m W;

    // Vector containing the singular values
    Vec_v sigma;

private:
    // Perform the singular value decomposition given the svd_type: Power, Jacobi, Parallel Jacobi
    void perform_SVD(Mat_m &A, Mat_m &U, Vec_v &sigma, Mat_m &V, const int r, const int svd_type);

    // Naive POD Algorithm, it only performs the SVD on the snapshot matrix S
    std::tuple<Mat_m, Vec_v> naive_POD(Mat_m &S, const int r, const int svd_type);

    // Algorithm 6.1 page 126 – POD Algorithm
    std::tuple<Mat_m, Vec_v> standard_POD(Mat_m &S, const int r, double const tol, const int svd_type);

    // Algorithm 6.2 page 128 – POD Algorithm with energy norm
    std::tuple<Mat_m, Vec_v> energy_POD(Mat_m &S, Mat_m &Xh, const int r, const double tol, const int svd_type);

    // Algorithm 6.3 page 134 – POD Algorithm with energy norm and quadrature weights
    std::tuple<Mat_m, Vec_v> weight_POD(Mat_m &S, Mat_m &Xh, Mat_m &D, const int r, const double tol, const int svd_type);

    // Standard incremental SVD for building POD – Algorithm 1
    // void standard_iSVD(Mat_m &U, Mat_m &Sigma, Mat_m &V, const Vec_v c, const double tol, const double tol_sv);

    // Enhanced incremental SVD for building POD – Algorithm 2
    // void enhanced_iSVD(Mat_m &U, Mat_m &Sigma, Mat_m &V, const Vec_v c, const int M, const double tol, const double tol_sv);
};

#endif