#ifndef SVD_HPP
#define SVD_HPP

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>
#include <mpi.h>

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;


class SVD {


public:
    

    SVD( Mat& data, bool full_matrices = true) : my_data(data), full_matrices_(full_matrices) {}
    
    virtual void compute();

    virtual Eigen::MatrixXd getV() const;

    virtual  Eigen::VectorXd getS() const;

    Eigen::MatrixXd getU() const;

    void setData(const Mat& A);

    void setU(const Mat& U_);

    void rSVD(int &target_rank,int &mpi_rank, int &mpi_size);

    void JacobiSVD(Mat &A);

private:
    void PM(Mat& A, Mat& B, double& sigma, Vec& u, Vec& v);

    void intermediate_step(Mat& A, Mat& B, Mat& Omega, int& l, int& q);

    void givens_rotation(double& a, double& b, Eigen::Matrix2d& G);

    void Jacobi_Rotation(Mat& A, double& s, double& tau, int& i, int& j, int& k, int& l);

    void qr_decomposition_full(const Mat& A, Mat& Q, Mat& R);

    void qr_decomposition_reduced(const Mat& A, Mat& Q, Mat& R);
    
    Mat my_data;
    Mat U;
    Vec S;
    Mat V;
protected:
    bool full_matrices_;
    
};

#endif // SVD_HPP