#include "rSVD.hpp"
#include "SVD.hpp"
#include "QR.hpp"
#include <mpi.h>

void intermediate_step(Mat_m &A, Mat_m &Q, Mat_m &Omega, int &l, int &q){
    
    Mat_m Y0 = A * Omega; // Y0 = A * Omega = (m*n) * (n*l) = (m*l)
    Mat_m Q0(A.rows(), l); // Q0 = (m*l)
    Mat_m R0(l, l); // R0 = (l*l)
    qr_decomposition_reduced(Y0, Q0, R0); 

    Mat_m Ytilde(A.cols(), l); // Ytilde = (n*l)
    Mat_m Qtilde(A.cols(), l); // Qtilde = (n*l)
    // It is useless to initialize Rtilde because it is still (l*l) and it can be overwritten
    
    for (int j = 1; j <= q; j++) {
        Ytilde = A.transpose() * Q0; // Y0 = A.transpose() * Q0 = (n*m) * (m*l) = (n*l)
        
        qr_decomposition_reduced(Ytilde, Qtilde, R0);

        Y0 = A * Qtilde; // Y0 = A * Qtilde = (m*n) * (n*l) = (m*l)
        
        qr_decomposition_reduced(Y0, Q0, R0);
        
    }
    Q = Q0;
}


 void rSVD(Mat_m& A, Mat_m& U, Vec_v& S, Mat_m& V, int l) {
    // Stage A
    // (1) Form an n × (k + p) Gaussian random matrix Omega
    int m = A.rows();
    int n = A.cols();

    // int k = 10; // numerical rank (we need an algorithm to find it) or target rank
    // int p = 5; // oversampling parameter, usually it is set to 5 or 10
    // int l = k + p;

    Mat_m Omega = Mat_m::Zero(n, l);

    // Create a random number generator for a normal distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    // Fill the corresponding rows of Omega with random numbers using this thread's seed
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < l; ++j) {
            Omega(i, j) = dist(gen);
        }
    }

    int q=2;
    Mat_m Q = Mat_m::Zero(m, l);
    intermediate_step(A, Q, Omega, l, q);
    // cout << "Q: " << Q << endl;
    // Stage B
    // (4) Form the (k + p) × n matrix B = Q*A
    Mat_m B = Q.transpose() * A;

    // (5) Form the SVD of the small matrix B
    int min= B.rows() < B.cols() ? B.rows() : B.cols();
    Mat_m Utilde = Mat_m::Zero(B.rows(), min);
    
    SVD(B, S, Utilde, V, min);

    // (6) Form U = Q*U_hat
    U = Q * Utilde;
}




