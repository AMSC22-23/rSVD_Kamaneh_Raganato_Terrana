#include "../include/powerMethod/PM1_omp.hpp"

// Matrix vector multiplication
void mat_vet_multiply(Mat &B, Vec &x0, Vec &res) { // other is a (cols*1) vector
    // Check if matrix and vector can be multiplied
    if (B.cols() != x0.size()) {
        std::cerr << "Error: Matrix dimensions are not compatible for multiplication.\n";
    }

    for (int i = 0; i < B.rows(); ++i) {
        double sum = 0.0;
        for (int k = 0; k < B.cols(); ++k) {
            sum += B(i, k) * x0[k];
        }
        res[i] = sum;
    }
}


// Matrix vector multiplication with OpenMP
void mat_vet_multiply_omp(Mat &B, Vec &x0, Vec &res, int num_threads) { // other is a (cols*1) vector
    // Check if matrix and vector can be multiplied
    if (B.cols() != x0.size()) {
        std::cerr << "Error: Matrix dimensions are not compatible for multiplication.\n";
    }

    omp_set_num_threads(num_threads);

    #pragma omp parallel for
    for (int i = 0; i < B.rows(); ++i) {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum) // NOT SURE
        for (int k = 0; k < B.cols(); ++k) {
            sum += B(i, k) * x0[k];
        }
        res[i] = sum;
    }
}


void PM(Mat &A, Mat &B, double &sigma, Vec &u, Vec &v) {
    // Generate a random initial guess x0
    Vec x0 = Vec::Zero(A.cols()); // To multiply B with x0

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> distribution(0.0, 1.0);

    for (int i=0; i<x0.size(); i++) {
        x0(i) = distribution(gen);
    }
    x0.normalize();
    // cout << "Check norm of x0: " << x0.norm() << endl;

    // Define the number of iterations
    double epsilon = 1.e-10;
    double delta = 0.05;
    double lambda = 0.1;
    int s = ceil( log(4*log(2*A.cols()/delta)/(epsilon*delta)) /(2*lambda));
    // cout << "Check the number of iterations: " << s << endl;

    // Check if the result of mat_vet_multiply is the same obtained with Eigen
    Vec x1 = x0;
    Vec x3 = x0;
    // cout << "\nCheck x1 before Eigen *: " << endl << x1 << endl;
    // cout << "\nCheck x0 before mat_vet_multiply: " << endl << x0 << endl;

    // Eigen matrix vector multiplication
    const auto t0 = high_resolution_clock::now();
    for (int i=1; i<=s; i++) {
        x1 = B*x1; // B = A^T*A
        x1.normalize();
    }
    const auto t1 = high_resolution_clock::now();
    const auto dt_eigen = duration_cast<milliseconds>(t1 - t0).count();
    std::cout << "\nElapsed mm eigen: " << dt_eigen << " [ms]\n";

    // Sequential matrix vector multiplication
    const auto t2 = high_resolution_clock::now();
    for (int i=1; i<=s; i++) {
        // x0 = B*x0; // B = A^T*A
        Vec x2 = x0; // Auxiliary variable
        mat_vet_multiply(B, x2, x0);
        x0.normalize();
    }
    const auto t3 = high_resolution_clock::now();
    const auto dt_sequential = duration_cast<milliseconds>(t3 - t2).count();
    std::cout << "Elapsed mm mat_vet_multiply: " << dt_sequential << " [ms]\n";
    cout << "Are x1 and x0 equal? ||x1-x0|| = " << (x1-x0).norm() << endl;
    // "Are x1 and x0 equal? ||x1-x0|| = 1.48174e-16" is printed at the first iteration of PM (called from SVD)
    // The norm of the difference between the vector computed with Eigen and the one computed with mat_vet_multiply is of the order of 1e-16

    // cout << "\nCheck x1 after Eigen *: " << endl << x1 << endl;
    // cout << "\nCheck x0 after mat_vet_multiply: " << endl << x0 << endl;

    // Parallel matrix vector multiplication
    const auto t4 = high_resolution_clock::now();
    for (int i=1; i<=s; i++) {
        // x0 = B*x0; // B = A^T*A
        Vec x4 = x3; // Auxiliary variable
        mat_vet_multiply_omp(B, x4, x3, 4);
        x3.normalize();
    }
    const auto t5 = high_resolution_clock::now();
    const auto dt_parallel = duration_cast<milliseconds>(t5 - t4).count();
    std::cout << "Elapsed mm mat_vet_multiply_omp: " << dt_parallel << " [ms]\n";
    cout << "Are x3 and x0 equal? ||x3-x0|| = " << (x3-x0).norm() << endl;

    // Compute the left singlular vector
    v = x0;
    v.normalize();

    // Compute the singular value
    sigma = (A*v).norm();

    // Compute the right singular vector
    u = A*v/sigma;

}

/*
void SVD(Mat &A, Vec &sigma, Mat &U, Mat &V) {
    Mat VT = Mat::Zero(A.cols(), A.cols()); // VT is the transpose of V

    // Define the matrix B = A^T*A
    Mat B = A.transpose()*A;

    // Define auxiliary vectors u and v
    Vec u = Vec::Zero(A.rows());
    Vec v = Vec::Zero(A.cols());
    
    for (int i=0; i<A.cols(); i++) {
        u = U.col(i);
        v = VT.row(i);
        PM(A, B, sigma(i), u, v);
        A -= sigma(i)*u*v.transpose();
        B = A.transpose()*A;
    }

    V = VT.transpose(); // VT is the transpose of V
}
*/


