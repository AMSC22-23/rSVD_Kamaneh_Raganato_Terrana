#ifndef POD_H
#define POD_H

#include <iostream>
#include <random>
#include <limits>
#include <numeric>
#include <tuple>

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

class POD
{
public:
    // Default constructor for standard POD
    POD(Mat_m &S, const int r, const double tol);

    // Default constructor for standard POD
    POD(Mat_m &S, const int r, const double tol);

    // Constructor for energy POD
    POD(Mat_m &S, Mat_m &Xh, const int r, const double tol);

    // Constructor for weight POD
    POD(Mat_m &S, Mat_m &Xh, Mat_m &D, const int r, const double tol);

    // Constructor for online POD through incremental SVD: starting from A it computes U, Sigma, V
    POD(Mat_m &A, Mat_m &U, Mat_m &Sigma, Mat_m &V, const int dim, Vec_v c, const int M, const int r, const double tol, const double tol_sv);

    // CAPIRE CHE TIPOLOGIA DI PRIVACY
    // STAMPA Q PER VEDERE SE ESCE GIUSTA, FATTI UN MINI DATASET DA CUI PARTIRE
    // SI PUÒ FARE ULTIMA RIGA CON LA NOSTRA QR?
    // POI MAGRI CAMBIA ANCHE I NOMI CHE STAI DANDO

    // CONTROLLA STAMPANDO SE conservativeResize FUNZIONA

    // Power Method
    void PM(Mat_m &A, Mat_m &B, double &sigma, Vec_v &u, Vec_v &v);

    // Singular Value Decomposition through Power Method
    void SVD(Mat_m &A, Vec_v &sigma, Mat_m &U, Mat_m &V, const int dim);

    // Algorithm 6.1 page 126 – POD Algorithm
    std::tuple<Mat_m, Vec_v> standard_POD(Mat_m &S, const int r, double const tol);

    // Algorithm 6.2 page 128 – POD Algorithm with energy norm
    std::tuple<Mat_m, Vec_v> energy_POD(Mat_m &S, Mat_m &Xh, const int r, const double tol);

    // Algorithm 6.3 page 134 – POD Algorithm with energy norm and quadrature weights
    std::tuple<Mat_m, Vec_v> weight_POD(Mat_m &S, Mat_m &Xh, Mat_m &D, const int r, const double tol);

    // Algorithm 6.4 page 137




    // INIZIALMENTE void POI MAGARI CAMBIA ESPLICITANDO OUTPUT
    // Standard incremental SVD for building POD – Algorithm 1
    void standard_iSVD(Mat_m &U, Mat_m &Sigma, Mat_m &V, const Vec_v c, const double tol, const double tol_sv);

    // Enhanced incremental SVD for building POD – Algorithm 2
    void enhanced_iSVD(Mat_m &U, Mat_m &Sigma, Mat_m &V, const Vec_v c, const int M, const double tol, const double tol_sv);

    // Matrice contenente modes
    Mat_m W;

    // Vector containing singular values
    Vec_v sigma;

};

#endif