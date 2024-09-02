
#ifndef rSVD_H
#define rSVD_H

#include <Eigen/Dense>
#include <Eigen/Sparse>


using Mat_m = Eigen::MatrixXd;
using Vec_v = Eigen::VectorXd;


void intermediate_step(Mat_m &A,Mat_m &Q,Mat_m &Omega,int &l,int &q);
void rSVD(Mat_m& A, Mat_m&U, Vec_v& S, Mat_m& V, int l);


#endif