#ifndef PM_H
#define PM_H

#include <iostream>
#include <fstream>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

using namespace std;
using namespace Eigen;

using Mat_m = MatrixXd;
using Vec_v = VectorXd;
using SpMat = SparseMatrix<double>;

void PM(Mat_m &A, Mat_m &B, double &sigma, Vec_v &u, Vec_v &v);

#endif


