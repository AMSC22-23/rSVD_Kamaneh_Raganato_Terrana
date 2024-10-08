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
#include <omp.h>
#include "JacobiOperations.hpp"
#include "PM.hpp"
#include "Jacobi_Class.hpp"
#include <mpi.h>

// Alias to simplify matrix and vector declarations
using Mat_m = Eigen::MatrixXd;
using Vec_v = Eigen::VectorXd;

// Enum to select the SVD method to use
enum class SVDMethod {
    Jacobi,
    Power,
    ParallelJacobi
};

// Template class for Singular Value Decomposition (SVD)
template<SVDMethod method>
class SVD {
public:
    // Constructor that takes the matrix to be decomposed
    // SVD(const Mat_m& data);
    SVD(const Mat_m& data, const int &r = 0);
    
    // Main method to compute the SVD
    void compute();

    // Methods to get the resulting U, S, V matrices
    Mat_m getU() const { return U_; }
    Vec_v getS() const { return S_; }
    Mat_m getV() const { return V_; }

private:
    // Internal matrices and vectors to store the decomposition results
    Mat_m U_;
    Vec_v S_;
    Mat_m V_;
    Mat_m data_;
    int r_;
    
    // Jacobi method implementation for SVD
    void jacobiSVD() ;

    // Power method implementation for SVD
    void powerMethodSVD() ;

    // Parallel Jacobi method implementation for SVD
    void ParallelJacobiSVD() ;

protected:
    void setData(const Mat_m& data) {
        data_ = data;
    }
};

template<SVDMethod method>
SVD<method>::SVD(const Mat_m& data, const int &r)
    : data_(data), r_(r) {}

// Main method to compute the SVD
template<SVDMethod method>
void SVD<method>::compute() {
    std::cout << "Entering compute() method" << std::endl;
    
    U_=Mat_m::Identity(data_.rows(), data_.rows());
    V_=Mat_m::Identity(data_.cols(), data_.cols());
    S_=Vec_v::Zero(std::min(data_.rows(), data_.cols()));

    if constexpr (method == SVDMethod::Jacobi) {
        std::cout << "Using Jacobi method" << std::endl;
        jacobiSVD();
    } else if constexpr (method == SVDMethod::Power) {
        std::cout << "Using Power method" << std::endl;
        powerMethodSVD();
    } 
    else if constexpr (method == SVDMethod::ParallelJacobi) {
        std::cout << "Using Parallel Jacobi method" << std::endl;
        ParallelJacobiSVD();
    }
}

// Jacobi method implementation for SVD
template<SVDMethod method>
void SVD<method>::jacobiSVD()  {
    size_t m = data_.rows();
    size_t n = data_.cols();
    size_t min_dim = std::min(m, n);
    // Initialize the working matrices
    Mat_m m_workMatrix = data_;
    U_ = Mat_m::Identity(m, min_dim);
    V_ = Mat_m::Identity(n, min_dim);
    
    if (m > n) {
        Mat_m B = m_workMatrix;
        Eigen::HouseholderQR<Mat_m> qr(B);
        m_workMatrix = qr.matrixQR().block(0, 0, n, n).triangularView<Eigen::Upper>();
        U_ = qr.householderQ() * U_;
        V_.setIdentity(n,n);
    }else if (n > m) {
    Mat_m m_adjoint = data_.adjoint();
    Eigen::HouseholderQR<Mat_m> m_qr(m_adjoint);
    m_workMatrix = m_qr.matrixQR().block(0, 0, m, m).triangularView<Eigen::Upper>().adjoint();
    U_.setIdentity(m,m);
    V_.setIdentity(n,m);
    V_ =m_qr.householderQ() * V_; 
    }
    
    bool finished = false;
    const double considerAsZero = (std::numeric_limits<double>::min)();
    const double precision = 2.0 * std::numeric_limits<double>::epsilon();
    double maxDiagEntry = m_workMatrix.cwiseAbs().diagonal().maxCoeff();
    double c_left = 0, s_left = 0, c_right = 0, s_right = 0;
   
    // Main loop for the Jacobi method
    while (!finished) {
        finished = true;
        for (size_t p = 1; p < min_dim; ++p) {
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
    for (size_t i = 0; i < min_dim; ++i) {
        double a = m_workMatrix(i, i);
        S_(i) = std::abs(a);
        if (a < 0) U_.col(i) = -U_.col(i);
    }

    size_t m_nonzeroSingularValues = min_dim;
    for(size_t i = 0; i < min_dim; i++) {
        size_t pos;
        double maxRemainingSingularValue = S_.tail(min_dim-i).maxCoeff(&pos);
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
    unsigned int dim;
    if (r_ == 0)
        dim = data_.cols() < data_.rows() ? data_.cols() : data_.rows();
    else
        dim = r_;
    
    Vec_v u = Vec_v::Zero(data_.rows());
    Vec_v v = Vec_v::Zero(data_.cols());
    Mat_m B = data_.transpose() * data_;

    for (unsigned int i = 0; i < dim; i++) {
        double sigma;
        PM(data_, B, sigma, u, v);
        if (sigma < 1e-12) {
            if (i == 0) {
                U_ = Mat_m::Zero(data_.rows(), 1);
                S_ = Vec_v::Zero(1);
                V_ = Mat_m::Zero(data_.cols(), 1);
            } else {
                U_.conservativeResize(data_.rows(), i);
                S_.conservativeResize(i);
                V_.conservativeResize(data_.cols(), i);
            }
            return;
        }
        Mat_m update = sigma * u * v.transpose();
        data_ -= update;
        B -= update.transpose() * update;
        U_.col(i) = u;
        V_.row(i) = v;
        S_(i) = sigma;
        
    }
    std::cout<<"V rows: "<<V_.rows()<<" V cols: "<<V_.cols()<<std::endl;
}


// Parallel Jacobi method implementation for SVD
template<SVDMethod method>
void SVD<method>::ParallelJacobiSVD()  {
    size_t m = data_.rows();
    size_t n = data_.cols();
    size_t min_dim = std::min(m, n);

    // Step 1: B = A
    Mat_m m_workMatrix = data_;

    // Step 2: U = I_mxn
    U_ = Mat_m::Identity(m, min_dim);

    // Step 3: V = I_nxn
    V_ = Mat_m::Identity(n, min_dim);

    if (m > n) {
        Mat_m B = m_workMatrix;
        Eigen::HouseholderQR<Mat_m> qr(B);
        m_workMatrix = qr.matrixQR().block(0, 0, n, n).triangularView<Eigen::Upper>();
        U_ = qr.householderQ() * U_;
    }else if (n > m) {
    Mat_m m_adjoint = data_.adjoint();
    Eigen::HouseholderQR<Mat_m> m_qr(m_adjoint);
    m_workMatrix = m_qr.matrixQR().block(0, 0, m, m).triangularView<Eigen::Upper>().adjoint();
    U_.setIdentity(m,m);
    V_.setIdentity(n,m);
    V_ =m_qr.householderQ() * V_; 
    }

    bool finished = false;
    const double considerAsZero = 1e-12; // Tolerance for zero
    const double precision = 1e-12;      // Precision tolerance
    double maxDiagEntry = m_workMatrix.cwiseAbs().diagonal().maxCoeff();
    double c_left = 0, s_left = 0, c_right = 0, s_right = 0;

    std::vector<std::tuple<double, size_t, size_t>> off_diagonal_weights;
    off_diagonal_weights.reserve(min_dim * (min_dim - 1) / 2);  // Pre-allocate space

    while (!finished) {
    
        finished = true;
        off_diagonal_weights.clear();  // Clear instead of re-allocating
        
        #pragma omp parallel
        {
            std::vector<std::tuple<double, size_t, size_t>> local_weights;
            #pragma omp for nowait 
            for (size_t p = 1; p < min_dim; ++p) {
                for (size_t q = 0; q < p; ++q) {
                    double threshold = std::max(considerAsZero, precision * maxDiagEntry);
                    double weight = m_workMatrix(p, q) * m_workMatrix(p, q) + m_workMatrix(q, p) * m_workMatrix(q, p);
                    if (weight > threshold) {
                        finished=false;
                        local_weights.emplace_back(weight, p, q);
                    }
                }
            }
            #pragma omp critical
            off_diagonal_weights.insert(end(off_diagonal_weights), begin(local_weights), end(local_weights));
        }
        
        if (!off_diagonal_weights.empty()) {
            // Ordina i blocchi fuori diagonale per norma di Frobenius decrescente
             std::sort(off_diagonal_weights.begin(), off_diagonal_weights.end(), std::greater<>());

             // Esegui la SVD sui blocchi con maggiore norma di Frobenius
            
            for (const auto &[weight, p, q] : off_diagonal_weights) {
                 if (svd_precondition_2x2_block_to_be_real(m_workMatrix, p, q, maxDiagEntry)) {
                    
                     real_2x2_jacobi_svd_par(m_workMatrix, c_left, s_left, c_right, s_right, p, q);
                    applyOnTheLeft_par(m_workMatrix, p, q, c_left, s_left);
                     applyOnTheRight_par(U_, p, q, c_left, -s_left);
                     applyOnTheRight_par(m_workMatrix, p, q, c_right, s_right);
                     applyOnTheRight_par(V_, p, q, c_right, s_right);
                    

                     // Aggiorna il massimo coefficiente diagonale
                     maxDiagEntry = std::max(maxDiagEntry, std::max(std::abs(m_workMatrix(p, p)), std::abs(m_workMatrix(q, q))));
                }
            }
        }
    }
    

    // Ensure positive diagonal entries
    #pragma omp parallel for
    for (size_t i = 0; i < min_dim; ++i) {
        double a = m_workMatrix(i, i);
        S_(i) = std::abs(a);
        if (a < 0) U_.col(i) = -U_.col(i);
    }

    // Sort singular values in descending order and compute the number of nonzero singular values
    size_t m_nonzeroSingularValues = min_dim;
    
    for (size_t i = 0; i < min_dim; ++i) {
        size_t pos;
        double maxRemainingSingularValue = S_.tail(min_dim - i).maxCoeff(&pos);
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
