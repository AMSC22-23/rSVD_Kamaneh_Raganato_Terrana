# include "POD.hpp"

int main () {

    // Naive test for online POD through incremental SVD
    int n = 8;
    Mat A = Mat::Zero(n, n);                           
    for (int i=0; i<n; i++) {
        A.coeffRef(i, i) = 2.0;
	    if(i>0) A.coeffRef(i, i-1) = -1.0;
        if(i<n-1) A.coeffRef(i, i+1) = -1.0;	
    }

    Mat A_e = A;

    const int dim = std::min(A.rows(), A.cols());
    Mat U     = Mat::Zero(A.rows(), dim);
    Mat Sigma = Mat::Zero(dim, dim);
    Mat V     = Mat::Zero(A.cols(), dim);

    Mat U_e     = Mat::Zero(A.rows(), dim);
    Mat Sigma_e = Mat::Zero(dim, dim);
    Mat V_e     = Mat::Zero(A.cols(), dim);

    const Vec c = Vec::Ones(n);
    const int M = 2;
    const int r = 5;
    const double tol    = 1e-12;
    const double tol_sv = 1e-12;

    // POD pod_(A, U, Sigma, V, dim, c, M, r, tol, tol_sv);
    // pod_.standard_iSVD(U, Sigma, V, c, tol, tol_sv);

    // POD pod_e(A_e, U_e, Sigma_e, V_e, dim, c, M, r, tol, tol_sv);
    // pod_e.enhanced_iSVD(U_e, Sigma_e, V_e, c, M, tol, tol_sv);

    // Naive test for standard POD
    Mat S1 = Mat::Zero(n, n);                           
    for (int i=0; i<n; i++) {
        S1.coeffRef(i, i) = 2.0;
	    if(i>0) S1.coeffRef(i, i-1) = -1.0;
        if(i<n-1) S1.coeffRef(i, i+1) = -1.0;	
    }
    POD pod1(S1, r, tol);

    // Naive test for energy POD
    int Nh = 5;
    int ns = 8;
    Mat S2 = Mat::Zero(Nh, ns);
    for (int i=0; i<Nh; i++) {
        S2.coeffRef(i, i) = 2.0;
	    if(i>0) S2.coeffRef(i, i-1) = -1.0;
        if(i<Nh-1) S2.coeffRef(i, i+1) = -1.0;	
    }
    Mat S3 = S2; // Initialize S3 here to prevent changes of S2 inside the constructor

    Mat Xh2 = Mat::Zero(Nh, Nh);
    for (int i=0; i<Nh; i++) {
        Xh2.coeffRef(i, i) = 2.0;
	    if(i>0) Xh2.coeffRef(i, i-1) = -1.0;
        if(i<Nh-1) Xh2.coeffRef(i, i+1) = -1.0;	
    }
    Mat Xh3 = Xh2; // Initialize Xh3 here to prevent changes of Xh2 inside the constructor

    POD pod2(S2, Xh2, r, tol);

    // Naive test for weight POD
    Mat D = Mat::Zero(ns, ns);
    for (int i=0; i<ns; i++) {
        D.coeffRef(i, i) = 0.1;
    }
    POD pod3(S3, Xh3, D, r, tol);

    return 0;
}