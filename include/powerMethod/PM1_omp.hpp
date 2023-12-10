#ifndef PM_H
#define PM_H

#include <iostream>
#include <fstream>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;
using namespace Eigen;

using Mat = MatrixXd;
using Vec = VectorXd;
using SpMat = SparseMatrix<double>;

void mat_vet_multiply(Mat &B, Vec &x0, Vec &res);

void mat_vet_multiply_omp(Mat &B, Vec &x0, Vec &res, int num_threads);

void PM(Mat &A, Mat &B, double &sigma, Vec &u, Vec &v);

// void SVD(Mat &A, Vec &sigma, Mat &U, Mat &V);

#endif


