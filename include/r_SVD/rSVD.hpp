
#ifndef rSVD_H
#define rSVD_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <omp.h>


using Mat = Eigen::MatrixXd;
using Vet = Eigen::VectorXd;


void intermediate_step(Mat &A,Mat &Q,Mat &Omega,size_t &l,size_t &q);
void rSVD(Mat& A, Mat&U, Vet& S, Mat& V);


#endif