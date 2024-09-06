#include "Jacobi_Class.hpp"
#include <cmath>
#include <limits>

JacobiRotation::JacobiRotation() : c(1), s(0), mat(Eigen::MatrixXd::Identity(2, 2)) {}

JacobiRotation::JacobiRotation(double c, double s) : c(c), s(s), mat(Eigen::MatrixXd(2, 2)) {
    mat << c, s, 
           -s, c;
}

double JacobiRotation::getC() const {
    return c;
}

double JacobiRotation::getS() const {
    return s;
}

void JacobiRotation::setC(double c) {
    this->c = c;
}

void JacobiRotation::setS(double s) {
    this->s = s;
}

Eigen::Matrix2d JacobiRotation::getMatrix() const {
    Eigen::Matrix2d mat;
    mat << getC(), getS(), 
           -1*getS(), getC();
    return mat;
}

Eigen::Vector2d JacobiRotation::apply(const Eigen::Vector2d& vec) const {
    return getMatrix() * vec;
}

bool JacobiRotation::makeJacobi(double x, double y, double z) {
    double deno = 2 * std::abs(y);
    if (deno < std::numeric_limits<double>::min()) {
        setC(1.0);
        setS(0.0);
        return false;
    } else {
        double tau = (x - z) / deno;
        double w = std::sqrt(tau * tau + 1);
        double t;
        if (tau > 0) {
            t = 1 / (tau + w);
        } else {
            t = 1 / (tau - w);
        }
        double segno = t > 0 ? 1 : -1;
        double n = 1 / std::sqrt(t * t + 1);
        setS(-segno * (y / std::abs(y)) * std::abs(t) * n);
        setC(n);
        return true;
    }
}