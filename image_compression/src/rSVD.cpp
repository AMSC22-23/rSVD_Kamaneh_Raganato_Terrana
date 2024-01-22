#include "SVD.hpp"
#include "rSVD.hpp"
#include "QR.hpp"
#include <mpi.h>
#include <omp.h>

void intermediate_step(Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& Omega, int &l, int &q){
    using namespace std;
    using namespace Eigen;

    using Mat = MatrixXd;
    using Vec = VectorXd;

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

void intermediate_step_mpi(Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& Omega, int &l, int &q){
    using namespace std;
    using namespace Eigen;

    using Mat = MatrixXd;
    using Vec = VectorXd;
    
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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


 void rSVD(Eigen::MatrixXd& A, Eigen::MatrixXd& U, Eigen::VectorXd& S, Eigen::MatrixXd& V, int l) {
    using namespace std;
    using namespace Eigen;

    using Mat = MatrixXd;
    using Vec = VectorXd;

    // Stage A
    // (1) Form an n × (k + p) Gaussian random matrix Omega
    int m = A.rows();
    int n = A.cols();

    Mat Omega = Mat::Zero(n, l);

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

    int q=1;
    Mat Q = Mat::Zero(m, l);
    intermediate_step(A, Q, Omega, l, q);
    
    // Stage B
    // (4) Form the (k + p) × n matrix B = Q*A
    Mat B = Q.transpose() * A;

    // (5) Form the SVD of the small matrix B
    int min_dim= B.rows() < B.cols() ? B.rows() : B.cols();
    Mat Utilde = Mat::Zero(B.rows(), min_dim);
    singularValueDecomposition(B, S, Utilde, V, min_dim);

    // (6) Form U = Q*U_hat
    U = Q * Utilde;
}





 void rSVD_mpi(Eigen::MatrixXd& A, Eigen::MatrixXd& U, Eigen::VectorXd& S, Eigen::MatrixXd& V, int l) {
    using namespace std;
    using namespace Eigen;

    using Mat = MatrixXd;
    using Vec = VectorXd;

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Stage A
    // (1) Form an n × (k + p) Gaussian random matrix Omega
    int m = A.rows();
    int n = A.cols();

    Mat Omega = Mat::Zero(n, l);

    int rows_per_proc = Omega.rows() / num_procs;
    int rows_remainder = Omega.rows() % num_procs;
    int local_rows = (rank < rows_remainder) ? rows_per_proc + 1 : rows_per_proc;
    int offset = rank * rows_per_proc + std::min(rank, rows_remainder);

    // Create a random number generator for a normal distribution
    std::random_device rd;
    std::mt19937 gen(rd() + rank);
    std::normal_distribution<double> dist(0.0, 1.0);

    // Fill the corresponding rows of Omega with random numbers using this thread's seed
    Eigen::MatrixXd local_Omega = Omega.block(offset, 0, local_rows, Omega.cols());
    for (int i = 0; i < local_Omega.rows(); ++i) {
        for (int j = 0; j < l; ++j) {
            local_Omega(i, j) = dist(gen);
        }
    }

    std::vector<int> recvcounts(num_procs);
    std::vector<int> displacements(num_procs);

    // Compute displacements and recvcounts arrays
    for (int i = 0; i < num_procs; ++i) {
        recvcounts[i] = (i < rows_remainder) ? (rows_per_proc + 1)*l : rows_per_proc*l;
        displacements[i] = (i < rows_remainder) ? (l * (i * (rows_per_proc + 1))) : (l * (rows_remainder * (rows_per_proc + 1)) + (i - rows_remainder) * l * rows_per_proc);    
    }

    MPI_Gatherv(local_Omega.data(), l*local_rows, MPI_DOUBLE, Omega.data(), recvcounts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int q=1;
    Mat Q = Mat::Zero(m, l);
    intermediate_step_mpi(A, Q, Omega, l, q);
    
    // Stage B
    // (4) Form the (k + p) × n matrix B = Q*A
    Mat B = Q.transpose() * A;

    // (5) Form the SVD of the small matrix B
    int min_dim= B.rows() < B.cols() ? B.rows() : B.cols();
    Mat Utilde = Mat::Zero(B.rows(), min_dim);
    singularValueDecomposition_mpi(B, S, Utilde, V, min_dim);

    // (6) Form U = Q*U_hat
    U = Q * Utilde;
}




