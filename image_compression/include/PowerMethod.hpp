#ifndef POWER_METHOD_H
#define POWER_METHOD_H

#include <iostream>
#include <fstream>
#include <random>
#include <mpi.h>
#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

/**

 */

void powerMethod(Eigen::MatrixXd& A, Eigen::MatrixXd& B, double& sigma, Eigen::VectorXd& u, Eigen::VectorXd& v);

#endif
