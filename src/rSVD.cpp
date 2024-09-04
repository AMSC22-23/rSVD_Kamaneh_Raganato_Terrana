#include <random>
#include <stdexcept>
#include <mpi.h>
#include <vector>

#include "../include/rSVD.hpp"
#include "../include/SVD_class.hpp"
#include "../include/QR.hpp"


// Function to generate the Omega matrix using MPI
Eigen::MatrixXd generateOmega(int n, int k) {
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    Eigen::MatrixXd Omega = Eigen::MatrixXd::Constant(n, k, 0.0);

    int rows_per_proc = Omega.rows() / num_procs;
    int rows_remainder = Omega.rows() % num_procs;
    int local_rows = (rank < rows_remainder) ? rows_per_proc + 1 : rows_per_proc;
    int offset = rank * rows_per_proc + std::min(rank, rows_remainder);

    // Initialization of the random number generator on each process
    std::random_device rd;
    // Use the rank of the process as the seed
    std::mt19937 gen(rd() + rank);
    std::normal_distribution<double> dist(0.0, 1.0);

    // Generate local portion of the matrix Omega
    Eigen::MatrixXd local_Omega = Omega.block(offset, 0, local_rows, Omega.cols());
    for (int i = 0; i < local_Omega.rows(); ++i) {
        for (int j = 0; j < k; ++j) {
            local_Omega(i, j) = dist(gen);
        }
    }

    std::vector<int> recvcounts(num_procs);
    std::vector<int> displacements(num_procs);

    // Compute displacements and recvcounts arrays
    for (int i = 0; i < num_procs; ++i) {
        recvcounts[i] = (i < rows_remainder) ? (rows_per_proc + 1) * k : rows_per_proc * k;
        displacements[i] = (i < rows_remainder) ? (k * (i * (rows_per_proc + 1))) : (k * (rows_remainder * (rows_per_proc + 1)) + (i - rows_remainder) * k * rows_per_proc);
    }

    // Gather local results using MPI_Gatherv
    MPI_Gatherv(local_Omega.data(), k * local_rows, MPI_DOUBLE, Omega.data(), recvcounts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Broadcast the Omega matrix from the root process to all other processes
    MPI_Bcast(Omega.data(), n * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    return Omega;
}

void intermediate_step(const Mat_m& A, Mat_m& Q, const Mat_m& Omega, int l, int q) {
    // Implementation of the intermediate step function
    Mat_m Y = A * Omega;
    Eigen::HouseholderQR<Mat_m> qr(Y);
    Q = qr.householderQ() * Mat_m::Identity(Y.rows(), l);
    for (int i = 0; i < q; ++i) {
        Y = A.transpose() * Q;
        qr.compute(Y);
        Q = qr.householderQ() * Mat_m::Identity(Y.rows(), l);
        Y = A * Q;
        qr.compute(Y);
        Q = qr.householderQ() * Mat_m::Identity(Y.rows(), l);
    }
}

void rSVD(Mat_m& A, Mat_m& U, Vec_v& S, Mat_m& V, int l, SVDMethod method) {
    // Initialize MPI
   

    // Stage A
    // (1) Form an n × (k + p) Gaussian random matrix Omega
    int m = A.rows();
    int n = A.cols();

    Mat_m Omega = generateOmega(n, l);

    int q = 2;
    Mat_m Q = Mat_m::Zero(m, l);
    intermediate_step(A, Q, Omega, l, q);

    // Stage B
    // (4) Form the (k + p) × n matrix B = Q*A
    Mat_m B = (Q.transpose() * A);
    
    // (5) Form the SVD of the small matrix B
    int min = B.rows() < B.cols() ? B.rows() : B.cols();
    Mat_m Utilde =Mat_m::Zero(B.rows(), min);

    // Declare the SVD object
    switch (method) {
        case SVDMethod::Jacobi: {
            
            SVD<SVDMethod::Jacobi> svd(B);
            svd.compute();
            S = svd.getS();
            Utilde = svd.getU();
            V = svd.getV();
            break;
        }
        case SVDMethod::Power: {
            SVD<SVDMethod::Power> svd(B);
            svd.compute();
            S = svd.getS();
            Utilde = svd.getU();
            V = svd.getV();
            break;
        }
        case SVDMethod::DynamicJacobi: {
            SVD<SVDMethod::DynamicJacobi> svd(B);
            svd.compute();
            S = svd.getS();
            Utilde = svd.getU();
            V = svd.getV();
            break;
        }
        case SVDMethod::ParallelJacobi: {
            SVD<SVDMethod::ParallelJacobi> svd(B);
            svd.compute();
            S = svd.getS();
            Utilde = svd.getU();
            V = svd.getV();
            break;
        }
        default:
            throw std::invalid_argument("Unsupported SVD method");
    }
    
    // (6) Form U = Q*U_hat
    std::cout<<"U before"<<std::endl;
    U = Q * Utilde;
    std::cout<<"U after"<<std::endl;
    
    
    
}