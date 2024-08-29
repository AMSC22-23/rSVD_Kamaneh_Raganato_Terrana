#include "SVD.hpp"

void singularValueDecomposition_mpi(Eigen::MatrixXd& A, Eigen::VectorXd& sigma, Eigen::MatrixXd& U, Eigen::MatrixXd& V, const int dim) {
    using namespace std;
    using namespace Eigen;

    using Mat = MatrixXd;
    using Vec = VectorXd;

    Mat VT = Mat::Zero(dim, A.cols()); // VT is the transpose of V

    // Define the matrix B = A^T*A
    Mat B = A.transpose()*A; // n*n

    // Define auxiliary vectors u and v
    Vec u = Vec::Zero(A.rows());
    Vec v = Vec::Zero(A.cols());
    
    for (int i=0; i<dim; i++) {
        powerMethod_mpi(A, B, sigma(i), u, v);
        A -= sigma(i)*u*v.transpose();
        B = A.transpose()*A;
        U.col(i) = u;
        VT.row(i) = v;
    }

    V = VT.transpose(); // V is the transpose of VT
}

void singularValueDecomposition(Eigen::MatrixXd& A, Eigen::VectorXd& sigma, Eigen::MatrixXd& U, Eigen::MatrixXd& V, const int dim) {
    using namespace std;
    using namespace Eigen;

    using Mat = MatrixXd;
    using Vec = VectorXd;

    Mat VT = Mat::Zero(dim, A.cols()); // VT is the transpose of V

    // Define the matrix B = A^T*A
    Mat B = A.transpose()*A; // n*n

    // Define auxiliary vectors u and v
    Vec u = Vec::Zero(A.rows());
    Vec v = Vec::Zero(A.cols());
    
    for (int i=0; i<dim; i++) {
        powerMethod(A, B, sigma(i), u, v);
        A -= sigma(i)*u*v.transpose();
        B = A.transpose()*A;
        U.col(i) = u;
        VT.row(i) = v;
    }

    V = VT.transpose(); // V is the transpose of VT
}