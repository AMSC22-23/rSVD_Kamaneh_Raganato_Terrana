#ifndef SVD_H
#define SVD_H

#include "PowerMethod.hpp" 

/**

 */
void singularValueDecomposition_mpi(Eigen::MatrixXd& A, Eigen::VectorXd& sigma, Eigen::MatrixXd& U, Eigen::MatrixXd& V, const int dim);

#endif
