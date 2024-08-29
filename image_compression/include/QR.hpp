#ifndef QR_H
#define QR_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>


#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

template <typename Scalar>
class QRDecomposition {
public:
    QRDecomposition(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A);

    virtual void decompose(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Q,
                           Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& R) const = 0;

protected:
    void givens_rotation(Scalar a, Scalar b, Eigen::Matrix<Scalar, 2, 2>& G) const;

    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& getA() const;

private:
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A_;
};

template <typename Scalar>
class QRFullDecomposition : public QRDecomposition<Scalar> {
public:
    using QRDecomposition<Scalar>::getA;
    using QRDecomposition<Scalar>::givens_rotation;

    QRFullDecomposition(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A);

    void decompose(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Q,
                   Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& R) const override;
};

template <typename Scalar>
class QRReducedDecomposition : public QRDecomposition<Scalar> {
public:
    using QRDecomposition<Scalar>::getA;
    using QRDecomposition<Scalar>::givens_rotation;

    QRReducedDecomposition(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A);

    void decompose(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Q,
                   Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& R) const override;
};

template <typename Scalar>
class QRMPIDecomposition : public QRDecomposition<Scalar> {
public:
    using QRDecomposition<Scalar>::getA;
    using QRDecomposition<Scalar>::givens_rotation;

    QRMPIDecomposition(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A);

    void decompose(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Q,
                   Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& R) const override;
};

#endif // QR_H



// /**
//  * @brief Compute the Givens rotation matrix.
//  * @param a The first element of the vector to be rotated.
//  * @param b The second element of the vector to be rotated.
//  * @return The Givens rotation matrix.
//  */
// Eigen::MatrixXd givens_rotation(double a, double b);

// /**
//  * @brief Perform full QR decomposition of a matrix.
//  * @param A The matrix to be decomposed.
//  * @param Q The orthogonal matrix in the QR decomposition.
//  * @param R The upper triangular matrix in the QR decomposition.
//  */
// void qr_decomposition_full(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R);

// /**
//  * @brief Perform reduced QR decomposition of a matrix.
//  * @param A The matrix to be decomposed.
//  * @param Q The orthogonal matrix in the QR decomposition.
//  * @param R The upper triangular matrix in the QR decomposition.
//  */
// void qr_decomposition_reduced(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R);

// void qr_decomposition_reduced_mpi(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R); 

// #endif // QR_H
