#ifndef POD_H
#define POD_H

#include <iostream>
#include <random>
#include <limits>
#include <numeric>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

using namespace std;
using namespace Eigen;

using Mat = MatrixXd;
using Vec = VectorXd;

class POD
{
public:
    // Default constructor for standard POD
    POD(Mat &S, const int r, const double tol);

    // Constructor for energy POD
    POD(Mat &S, Mat &Xh, const int r, const double tol);

    // Constructor for weight POD
    POD(Mat &S, Mat &Xh, Mat &D, const int r, const double tol);

    // Constructor for online POD through incremental SVD: starting from A it computes U, Sigma, V
    POD(Mat &A, Mat &U, Mat &Sigma, Mat &V, const int dim, Vec c, const int M, const int r, const double tol, const double tol_sv);

    // CAPIRE CHE TIPOLOGIA DI PRIVACY
    // STAMPA Q PER VEDERE SE ESCE GIUSTA, FATTI UN MINI DATASET DA CUI PARTIRE
    // SI PUÒ FARE ULTIMA RIGA CON LA NOSTRA QR?
    // POI MAGRI CAMBIA ANCHE I NOMI CHE STAI DANDO

    // CONTROLLA STAMPANDO SE conservativeResize FUNZIONA

    // Power Method
    void PM(Mat &A, Mat &B, double &sigma, Vec &u, Vec &v);

    // Singular Value Decomposition through Power Method
    void SVD(Mat &A, Vec &sigma, Mat &U, Mat &V, const int dim);

    // Algorithm 6.1 page 126 – POD Algorithm
    Mat standard_POD(Mat &S, const int r, double const tol);

    // Algorithm 6.2 page 128 – POD Algorithm with energy norm
    Mat energy_POD(Mat &S, Mat &Xh, const int r, const double tol);

    // Algorithm 6.3 page 134 – POD Algorithm with energy norm and quadrature weights
    Mat weight_POD(Mat &S, Mat &Xh, Mat &D, const int r, const double tol);

    // Algorithm 6.4 page 137




    // INIZIALMENTE void POI MAGARI CAMBIA ESPLICITANDO OUTPUT
    // Standard incremental SVD for building POD – Algorithm 1
    void standard_iSVD(Mat &U, Mat &Sigma, Mat &V, const Vec c, const double tol, const double tol_sv);

    // Enhanced incremental SVD for building POD – Algorithm 2
    void enhanced_iSVD(Mat &U, Mat &Sigma, Mat &V, const Vec c, const int M, const double tol, const double tol_sv);

};

#endif