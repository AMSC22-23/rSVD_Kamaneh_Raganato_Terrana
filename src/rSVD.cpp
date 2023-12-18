#include "../include/r_SVD/rSVD.hpp"
#include "../include/powerMethod/SVD.hpp"
#include "../include/QRdecomposition/QR.hpp"

void intermediate_step(Mat &A, Mat &Q, Mat &Omega, int &l, int &q){
    
    Mat Y0 = A * Omega; // Y0 = A * Omega = (m*n) * (n*l) = (m*l)
    Mat Q0(A.rows(), l); // Q0 = (m*l)
    Mat R0(l, l); // R0 = (l*l)
    qr_decomposition_reduced(Y0, Q0, R0); 

    Mat Ytilde(A.cols(), l); // Ytilde = (n*l)
    Mat Qtilde(A.cols(), l); // Qtilde = (n*l)
    // It is useless to initialize Rtilde because it is still (l*l) and it can be overwritten
    
    for (int j = 1; j <= q; j++) {
        Ytilde = A.transpose() * Q0; // Y0 = A.transpose() * Q0 = (n*m) * (m*l) = (n*l)
        
        qr_decomposition_reduced(Ytilde, Qtilde, R0);

        Y0 = A * Qtilde; // Y0 = A * Qtilde = (m*n) * (n*l) = (m*l)
        
        qr_decomposition_reduced(Y0, Q0, R0);
        
    }
    Q = Q0;
}


 void rSVD(Mat& A, Mat& U, Vet& S, Mat& V) {
    // Stage A
    // (1) Form an n × (k + p) Gaussian random matrix Omega
    int m=A.rows();
    int n=A.cols();

    int k = 10; // numerical rank (we need an algorithm to find it) or target rank
    int p = 5; // oversampling parameter, usually it is set to 5 or 10
    int l = k + p;

    Mat Omega = Mat::Zero(n, l);

    // Create a random number generator for a normal distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);

    // Fill the matrix with values from a standard normal distribution
    for (int i = 0; i < Omega.rows(); ++i) {
        for (int j = 0; j < Omega.cols(); ++j) {
            Omega(i, j) = distribution(gen);
        }
    }
    int q=2;
    
    Mat Q = Mat::Zero(m, l);
    intermediate_step(A, Q, Omega, l, q);
    
    // Stage B
    // (4) Form the (k + p) × n matrix B = Q*A
    Mat B = Q.transpose() * A; // B = Q.transpose() * A = (l*m) * (m*n) = (l*n)

    // (5) Form the SVD of the small matrix B
    int min= B.rows() < B.cols() ? B.rows() : B.cols();
    Mat Utilde = Mat::Zero(B.rows(), min);
    //std::cout << "Utilde = \n" << Utilde << std::endl;
    SVD(B, S, Utilde, V, min);

    // // (6) Form U = Q*U_hat
    U = Q * Utilde; // U = Q * Utilde = (m*l) * (l*l) = (m*l) assuming l<n
}




