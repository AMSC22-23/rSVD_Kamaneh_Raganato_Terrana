#include "QR.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank==0) std::cout << "*** QR test 2 ***\n" << std::endl;

    Eigen::MatrixXd A(4, 3);

    A << 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0;

    int m = A.rows();
    int n = A.cols();
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(m, std::min(m, n));
    Eigen::MatrixXd R = A.topRows(std::min(m, n));
    QRReducedDecomposition<double> qrReduced(A);
    qrReduced.decompose(Q, R);
    
    if (rank == 0){
        cout << "Q: \n" << Q << endl;
        cout << "R: \n" << R << endl;

        Eigen::MatrixXd A_tilde = Q * R;
        cout << "A~ :\n" << A_tilde << endl;
    }

    MPI_Finalize();
    return 0;
}