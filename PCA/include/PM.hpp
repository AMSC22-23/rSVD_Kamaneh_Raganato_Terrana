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

using Mat = MatrixXd;
using Vec = VectorXd;
using SpMat = SparseMatrix<double>;

void PM(Mat &A, Mat &B, double &sigma, Vec &u, Vec &v);

#endif


