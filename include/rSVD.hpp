
#ifndef rSVD_H
#define rSVD_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "SVD_class.hpp"

using Mat_m = Eigen::MatrixXd;
using Vec_v = Eigen::VectorXd;


void intermediate_step(const Mat_m &A, Mat_m &Q, const Mat_m &Omega, int l, int q);
void rSVD(Mat_m &A, Mat_m &U, Vec_v &S, Mat_m &V, int l, SVDMethod method);
Mat_m generateOmega(int n, int l);


#endif