#include "SVD.hpp"

void SVD(Mat &A, Vec &sigma, Mat &U, Mat &V, const int dim) {

    Mat VT = Mat::Zero(dim, A.cols()); // VT is the transpose of V

    // Define the matrix B = A^T*A
    Mat B = A.transpose()*A; // n*n

    // Define auxiliary vectors u and v
    Vec u = Vec::Zero(A.rows());
    Vec v = Vec::Zero(A.cols());
    
    for (int i=0; i<dim; i++) {
        PM(A, B, sigma(i), u, v);
        A -= sigma(i)*u*v.transpose();
        B = A.transpose()*A;
        U.col(i) = u;
        VT.row(i) = v;
    }

    V = VT.transpose(); // V is the transpose of VT
}