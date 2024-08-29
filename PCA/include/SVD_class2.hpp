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


class SVD {


public:
    using Mat = Eigen::MatrixXd;
    using Vec = Eigen::VectorXd;

    SVD( Eigen::MatrixXd& my_dataa, bool full_matrices = true) : my_data(my_dataa), full_matrices_(full_matrices) {}
    
    virtual void compute();

    Eigen::MatrixXd getU() const;

    Eigen::VectorXd getS() const;

    Eigen::MatrixXd getV() const;

    void setData(const Mat& A);

    void setU(const Mat& U__);

    void rSVD(int &l);

private:
    void PM(Mat& A, Mat& B, double& sigma, Vec& u, Vec& v);

    void intermediate_step(Mat& A, Mat& B,Mat &Omega, int &l,int &q);

    void givens_rotation(double a, double b, Eigen::Matrix2d& G);

    void qr_decomposition_full(const Mat &A, Mat &Q, Mat &R);

    void qr_decomposition_reduced(const Mat &A, Mat &Q, Mat &R);
    
    Mat my_data;
    Mat U;
    Vec S;
    Mat V;
    bool full_matrices_;
    
};

#endif // SVD_HPP