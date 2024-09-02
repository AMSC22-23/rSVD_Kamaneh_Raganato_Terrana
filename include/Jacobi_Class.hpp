#ifndef JACOBIROTATION_H
#define JACOBIROTATION_H

#include <Eigen/Dense>
#include <iostream> 
using Mat_m=Eigen::MatrixXd;
using Vec_v=Eigen::VectorXd;
class JacobiRotation {
private:
    double c, s;
    Eigen::MatrixXd mat;

public:
    JacobiRotation();
    JacobiRotation(double c, double s);

    double getC() const;
    double getS() const;

    void setC(double c);
    void setS(double s);

    Eigen::Matrix2d getMatrix() const;
    Eigen::Vector2d apply(const Eigen::Vector2d& vec) const;
    bool makeJacobi(double x, double y, double z);
    JacobiRotation transpose()  {
        JacobiRotation transposed;
        transposed.mat = this->mat.transpose();
        return transposed;
    }
    Mat_m operator*(const JacobiRotation& other) const  {
        return this->mat * other.mat;
    }
    public:
    void setMat(const Mat_m& newMat) {
        this->mat = newMat;
    }
    void printMatrix() const {
    Eigen::Matrix2d m = getMatrix();
    std::cout << m << std::endl;
}
};

#endif // JACOBIROTATION_H