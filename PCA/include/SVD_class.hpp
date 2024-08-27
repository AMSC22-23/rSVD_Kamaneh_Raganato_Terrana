#ifndef SVD_CLASS_HPP
#define SVD_CLASS_HPP

#include <Eigen/Dense>
#include <iostream>
#include <queue>
#include <cmath>
#include <limits>
#include <Eigen/QR>
#include <fstream>
#include <ctime>
#include <string>
#include <sstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include "JacobiOperations.hpp"
#include "PM.hpp"
#include "Jacobi_Class.hpp"
#include <mpi.h>

// Alias to simplify matrix and vector declarations
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

// Enum to select the SVD method to use
enum class SVDMethod {
    Jacobi,
    Power,
    DynamicJacobi
};

// Template class for Singular Value Decomposition (SVD)
template<SVDMethod method>
class SVD {
public:
    // Constructor that takes the matrix to be decomposed
    SVD(const Mat& data);
    
    // Main method to compute the SVD
    void compute();

    // Methods to get the resulting U, S, V matrices
    Mat getU() const { return U_; }
    Vec getS() const { return S_; }
    Mat getV() const { return V_; }

private:
    // Internal matrices and vectors to store the decomposition results
    Mat U_;
    Vec S_;
    Mat V_;
    Mat data_;
    
    // Jacobi method implementation for SVD
    void jacobiSVD() ;

    // Power method implementation for SVD
    void powerMethodSVD() ;

    // Dynamic Jacobi method implementation for SVD
    void DynamicJacobiSVD() ;
};

// Constructor implementation
template<SVDMethod method>
SVD<method>::SVD(const Mat& data)
    : data_(data) {}

// Main method to compute the SVD
template<SVDMethod method>
void SVD<method>::compute() {
    U_=Mat::Identity(data_.rows(), data_.rows());
    V_=Mat::Identity(data_.cols(), data_.cols());
    S_=Vec::Zero(std::min(data_.rows(), data_.cols()));

    if constexpr (method == SVDMethod::Jacobi) {
        jacobiSVD();
    } else if constexpr (method == SVDMethod::Power) {
        powerMethodSVD();
    } else if constexpr (method == SVDMethod::DynamicJacobi) {
        DynamicJacobiSVD();
    }
}

// Jacobi method implementation for SVD
template<SVDMethod method>
void SVD<method>::jacobiSVD()  {
    size_t m = data_.rows();
    size_t n = data_.cols();

    // Initialize the working matrices
    Mat m_workMatrix = data_;
    U_ = Mat::Identity(m, n);
    V_ = Mat::Identity(n, n);
    
    if (m > n) {
        Mat B = m_workMatrix;
        Eigen::HouseholderQR<Mat> qr(B);
        m_workMatrix = qr.matrixQR().block(0, 0, n, n).triangularView<Eigen::Upper>();
        U_ = qr.householderQ() * U_;
    }
    
    bool finished = false;
    const double considerAsZero = (std::numeric_limits<double>::min)();
    const double precision = 2.0 * std::numeric_limits<double>::epsilon();
    double maxDiagEntry = m_workMatrix.cwiseAbs().diagonal().maxCoeff();
    double c_left = 0, s_left = 0, c_right = 0, s_right = 0;
   
    // Main loop for the Jacobi method
    while (!finished) {
        finished = true;
        for (size_t p = 1; p < n; ++p) {
            for (size_t q = 0; q < p; ++q) {
                double threshold = std::max(considerAsZero, precision * maxDiagEntry);
                
                // Check if the 2x2 block is already diagonal
                if (std::abs(m_workMatrix(p,q)) > threshold || std::abs(m_workMatrix(q,p)) > threshold) {
                    finished = false;

                    // SVD of the 2x2 block
                    if (svd_precondition_2x2_block_to_be_real(m_workMatrix, p, q, maxDiagEntry)) {
                        real_2x2_jacobi_svd(m_workMatrix, c_left, s_left, c_right, s_right, p, q);
                        applyOnTheLeft(m_workMatrix, p, q, c_left, s_left);
                        applyOnTheRight(U_, p, q, c_left, -s_left);
                        applyOnTheRight(m_workMatrix, p, q, c_right, s_right);
                        applyOnTheRight(V_, p, q, c_right, s_right);

                        maxDiagEntry = std::max(maxDiagEntry, std::max(std::abs(m_workMatrix(p,p)), std::abs(m_workMatrix(q,q))));
                    }
                }
            }
        }
    }

    // Make the diagonal positive and sort the singular values
    for (size_t i = 0; i < m_workMatrix.rows(); ++i) {
        double a = m_workMatrix(i, i);
        S_(i) = std::abs(a);
        if (a < 0) U_.col(i) = -U_.col(i);
    }

    size_t m_nonzeroSingularValues = S_.size();
    for(size_t i = 0; i < S_.size(); i++) {
        size_t pos;
        double maxRemainingSingularValue = S_.tail(S_.size()-i).maxCoeff(&pos);
        if(maxRemainingSingularValue == 0) {
            m_nonzeroSingularValues = i;
            break;
        }
        if(pos) {
            pos += i;
            std::swap(S_.coeffRef(i), S_.coeffRef(pos));
            U_.col(pos).swap(U_.col(i));
            V_.col(pos).swap(V_.col(i));
        }
    }
    
}

// Power method implementation for SVD
template<SVDMethod method>
void SVD<method>::powerMethodSVD() {
    int dim = data_.cols() < data_.rows() ? data_.cols() : data_.rows();
    
    Vec u = Vec::Zero(data_.rows());
    Vec v = Vec::Zero(data_.cols());
    Mat B = data_.transpose() * data_;

    for (unsigned int i = 0; i < dim; i++) {
        double sigma;
        PM(data_, B, sigma, u, v);
        if (sigma < 1e-12) {
            if (i == 0) {
                U_ = Mat::Zero(data_.rows(), 1);
                S_ = Vec::Zero(1);
                V_ = Mat::Zero(data_.cols(), 1);
            } else {
                U_.conservativeResize(data_.rows(), i);
                S_.conservativeResize(i);
                V_.conservativeResize(data_.cols(), i);
            }
            return;
        }
        Mat update = sigma * u * v.transpose();
        data_ -= update;
        B -= update.transpose() * update;
        U_.col(i) = u;
        V_.row(i) = v;
        S_(i) = sigma;
    }
}

// Dynamic Jacobi method implementation for SVD
template<SVDMethod method>
void SVD<method>::DynamicJacobiSVD()  {
    size_t m = data_.rows();
    size_t n = data_.cols();
    

    // Initialize the working matrices
    Mat m_workMatrix = data_;
    U_ = Mat::Identity(m, n);
    V_ = Mat::Identity(n, n);
    
    if (m > n) {
        Mat B = m_workMatrix;
        Eigen::HouseholderQR<Mat> qr(B);
        m_workMatrix = qr.matrixQR().block(0, 0, n, n).triangularView<Eigen::Upper>();
        U_ = qr.householderQ() * U_;
    }
    
    bool finished = false;
    const double considerAsZero = (std::numeric_limits<double>::min)();
    const double precision = 2.0 *  std::numeric_limits<double>::epsilon();
    double maxDiagEntry = m_workMatrix.cwiseAbs().diagonal().maxCoeff();
    double c_left = 0, s_left = 0, c_right = 0, s_right = 0;

    std::vector<std::tuple<double, size_t, size_t>> off_diagonal_weights;
    off_diagonal_weights.reserve(n * (n - 1) / 2);  // Pre-allocate space

    // Main loop for the Dynamic Jacobi method
    while (!finished) {
        
        finished = true;
        off_diagonal_weights.clear();  // Clear instead of re-allocating
        
        for (size_t p = 1; p < n; ++p) {
            for (size_t q = 0; q < p; ++q) {
                double threshold = std::max(considerAsZero, precision * maxDiagEntry);

                // Calculate the Frobenius norm of the off-diagonal 2x2 block
                double weight = m_workMatrix(p, q) * m_workMatrix(p, q) +
                                m_workMatrix(q, p) * m_workMatrix(q, p);

                if (weight > threshold) {
                    off_diagonal_weights.emplace_back(weight, p, q);
                    finished = false; // Not finished yet, since there are elements above the threshold
                }
            }
        }
        
        if (!off_diagonal_weights.empty()) {
            // Sort the off-diagonal blocks by decreasing Frobenius norm
            std::sort(off_diagonal_weights.begin(), off_diagonal_weights.end(), std::greater<>());

            // Perform SVD on the blocks with the largest Frobenius norm
            for (const auto &[weight, p, q] : off_diagonal_weights) {
                if (svd_precondition_2x2_block_to_be_real(m_workMatrix, p, q, maxDiagEntry)) {
                    real_2x2_jacobi_svd(m_workMatrix, c_left, s_left, c_right, s_right, p, q);
                    applyOnTheLeft(m_workMatrix, p, q, c_left, s_left);
                    applyOnTheRight(U_, p, q, c_left, -s_left);
                    applyOnTheRight(m_workMatrix, p, q, c_right, s_right);
                    applyOnTheRight(V_, p, q, c_right, s_right);

                    // Update the maximum diagonal coefficient
                    maxDiagEntry = std::max(maxDiagEntry, std::max(std::abs(m_workMatrix(p, p)), std::abs(m_workMatrix(q, q))));
                }
            }
        }
    }
    
    // Make the diagonal positive and sort the singular values
    for (size_t i = 0; i < m_workMatrix.rows(); ++i) {
        double a = m_workMatrix(i, i);
        S_(i) = std::abs(a);
        if (a < 0) U_.col(i) = -U_.col(i);
    }

    size_t m_nonzeroSingularValues = S_.size();
    for (size_t i = 0; i < S_.size(); ++i) {
        size_t pos;
        double maxRemainingSingularValue = S_.tail(S_.size() - i).maxCoeff(&pos);
        if (maxRemainingSingularValue == 0) {
            m_nonzeroSingularValues = i;
            break;
        }
        if (pos) {
            pos += i;
            std::swap(S_.coeffRef(i), S_.coeffRef(pos));
            U_.col(pos).swap(U_.col(i));
            V_.col(pos).swap(V_.col(i));
        }
    }
    
}

#endif // SVD_CLASS_HPP
