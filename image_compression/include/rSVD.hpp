
#ifndef rSVD_H
#define rSVD_H

#include <Eigen/Dense>
#include <Eigen/Sparse>


void intermediate_step_mpi(Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& Omega, int &l, int &q);
void rSVD_mpi(Eigen::MatrixXd& A, Eigen::MatrixXd& U, Eigen::VectorXd& S, Eigen::MatrixXd& V, int l);

void intermediate_step(Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& Omega, int &l, int &q);
void rSVD(Eigen::MatrixXd& A, Eigen::MatrixXd& U, Eigen::VectorXd& S, Eigen::MatrixXd& V, int l);


#endif