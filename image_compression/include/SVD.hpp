#ifndef SVD_H
#define SVD_H

#include "PowerMethod.hpp" 

/**

 */
void singularValueDecomposition(Eigen::MatrixXd& A, Eigen::VectorXd& sigma, Eigen::MatrixXd& U, Eigen::MatrixXd& V, const int dim);

#endif