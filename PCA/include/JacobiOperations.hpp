#ifndef JACOBISVD_H
#define JACOBISVD_H



#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <limits>


using Mat=Eigen::MatrixXd;
using Vec=Eigen::VectorXd;


void applyOnTheLeft(Mat &matrix, int p, int q, double c, double s);
void applyOnTheRight(Mat &matrix, int p, int q, double c, double s);
void real_2x2_jacobi_svd(Mat &matrix, double &c_left,double &s_left,double &c_right,double &s_right,int p, int q);
bool svd_precondition_2x2_block_to_be_real(Mat& m_workMatrix, int p, int q, double maxDiagEntry);

// for the parallel version
std::vector<std::pair<size_t, size_t>> greedy_maximum_weight_matching(const std::vector<std::tuple<double, size_t, size_t>>& weights) ;


#endif // JACOBISVD_H