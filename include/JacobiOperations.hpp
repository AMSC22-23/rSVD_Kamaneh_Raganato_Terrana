#ifndef JACOBISVD_H
#define JACOBISVD_H



#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <limits>


using Mat_m=Eigen::MatrixXd;
using Vec_v=Eigen::VectorXd;


void applyOnTheLeft(Mat_m &matrix, int p, int q, double c, double s);
void applyOnTheRight(Mat_m &matrix, int p, int q, double c, double s);
void real_2x2_jacobi_svd(Mat_m &matrix, double &c_left,double &s_left,double &c_right,double &s_right,int p, int q);
bool svd_precondition_2x2_block_to_be_real(Mat_m& m_workMatrix, int p, int q, double maxDiagEntry);

// for the parallel version
void applyOnTheLeft_par(Mat_m &matrix, size_t p, size_t q, double c, double s);
void applyOnTheRight_par(Mat_m &matrix, size_t p, size_t q, double c, double s);
void real_2x2_jacobi_svd_par(Mat_m &matrix, double &c_left,double &s_left,double &c_right,double &s_right,size_t  p, size_t  q);


#endif // JACOBISVD_H