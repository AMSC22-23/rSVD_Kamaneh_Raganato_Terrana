#include "SVD.hpp"

// Power Method
void
SVD::PM(Mat_m &A, Mat_m &B, double &sigma, Vec_v &u, Vec_v &v)
{
    // Generate a random initial guess x0
    Vec_v x0 = Vec_v::Zero(A.cols());

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> distribution(0.0, 1.0);

    for (unsigned int i=0; i<x0.size(); i++) {
        x0(i) = distribution(gen);
    }
    x0.normalize();

    // Define the number of iterations
    double epsilon = 1.e-10;
    double delta = 0.05;
    double lambda = 0.1;
    int s = ceil( log(4*log(2*A.cols()/delta)/(epsilon*delta)) /(2*lambda));
    // cout << "Check the number of iterations: " << s << endl;

    for (unsigned int i=1; i<=s; i++) {
        x0 = B*x0; // B = A^T*A
        x0.normalize();
    }

    // Compute the left singlular vector
    v = x0;
    v.normalize();

    // Compute the singular value
    sigma = (A*v).norm();

    // Compute the right singular vector
    u = A*v/sigma;
}

// Singular Value Decomposition through Power Method
SVD::SVD(Mat_m &A, Vec_v &sigma, Mat_m &U, Mat_m &V, const int dim)
{
    Mat_m VT = Mat_m::Zero(dim, A.cols()); // VT is the transpose of V

    // Define the matrix B = A^T*A
    Mat_m B = A.transpose()*A; // n*n

    // Define auxiliary vectors u and v
    Vec_v u = Vec_v::Zero(A.rows());
    Vec_v v = Vec_v::Zero(A.cols());
    
    for (unsigned int i=0; i<dim; i++) {
        PM(A, B, sigma(i), u, v);
        A -= sigma(i)*u*v.transpose();
        B = A.transpose()*A;
        U.col(i) = u;
        VT.row(i) = v;
    }

    V = VT.transpose(); // V is the transpose of VT
}



