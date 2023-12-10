#include "../include/powerMethod/SVD_omp.hpp"

void SVD(Mat &A, Vec &sigma, Mat &U, Mat &V, const int dim) {
    // Mat VT = Mat::Zero(A.cols(), dim); // VT is the transpose of V

    // Define the matrix B = A^T*A
    Mat B = A.transpose()*A; // n*n

    // Define auxiliary vectors u and v
    Vec u = Vec::Zero(A.rows());
    Vec v = Vec::Zero(A.cols());
    
    for (int i=0; i<10; i++) { // It should be i<dim, here we use i<10 to print only the first 10 elapsed times
        u = U.col(i);
        // v = VT.row(i);
        // v = VT.col(i);
        // v = V.col(i);
        v = V.row(i);
        PM(A, B, sigma(i), u, v);
        A -= sigma(i)*u*v.transpose();
        B = A.transpose()*A;
        U.col(i) = u;
        // VT.row(i) = v;
        // VT.col(i) = v;
        // V.col(i) = v;
        V.row(i) = v;
    }

    // V = VT.transpose(); // VT is the transpose of V
}