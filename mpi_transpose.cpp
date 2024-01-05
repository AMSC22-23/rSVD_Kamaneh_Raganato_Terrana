#include <iostream>
#include <Eigen/Dense>
#include <mpi.h>
#include <chrono>

using namespace std;
using namespace Eigen;

using Mat = MatrixXd;
using Vec = VectorXd;

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the total number of processors and the rank of the current processor
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialize random seed
    srand(time(NULL));

    // Define the size of the matrix
    const int rows = 9;
    const int cols = 4;

    // Define a matrix A of size rows*cols and initialize with random values
    Mat A = Mat::Random(rows, cols);
    Mat AT_byrows = Mat::Random(cols, rows);

    // Divide rows of A among processors
    int rows_per_proc = A.rows() / num_procs;
    int rows_remainder = A.rows() % num_procs;
    int local_rows = (rank < rows_remainder) ? rows_per_proc + 1 : rows_per_proc;
    int offset = rank * rows_per_proc + std::min(rank, rows_remainder);

    Mat local_A_byrows = A.block(offset, 0, local_rows, A.cols());

    // Mat local_AT_byrows = Mat::Zero(A.cols(), local_rows);
    // Since to use MPI_Gatherv this matrix has to be flattened, it is better to use a vector from the beginning
    Vec local_AT_byrows_flatten = Vec::Zero(A.cols() * local_rows);

    // Check if the matrix A in divided correctly
    // For a 5*5 matrix, the first processor takes 3 rows, the second takes 2 rows
    cout << "Size of local_A_byrows:          " << local_A_byrows.rows() << "x" << local_A_byrows.cols() << endl;
    cout << "Size of local_AT_byrows_flatten: " << local_AT_byrows_flatten.size() << endl;
    // cout << "Size of local_AT_byrows: " << local_AT_byrows.rows() << "x" << local_AT_byrows.cols() << endl;
    // Size of local_A_byrows:  2x5
    // Size of local_A_byrows:  3x5
    // Size of local_AT_byrows: 5x3
    // Size of local_AT_byrows: 5x2

    // FOR A SQUARE MATRIX, IS IT BETTER TO DIVIDE AMONG PROCESSORS BY ROWS OR BY COLUMNS?
    // Divide rows of A among processors
    // int cols_per_proc = A.cols() / num_procs;
    // int cols_remainder = A.cols() % num_procs;
    // int local_cols = (rank < cols_remainder) ? cols_per_proc + 1 : cols_per_proc;
    // int offset = rank * cols_per_proc + std::min(rank, cols_remainder);

    // Mat local_A_bycols = A.block(0, offset, A.rows(), local_cols); CREDO

    // Transpose the local matrix local_A_byrows
    for (size_t i = 0; i < local_rows; ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            local_AT_byrows_flatten[i*A.cols()+j] = local_A_byrows(i, j);
        }
    }

    // Gather local results using MPI_Gatherv
    // Compute displacements and recvcounts arrays
    std::vector<int> recvcounts(num_procs); // Number of elements that each MPI process receives
    std::vector<int> displacements(num_procs); // Starting index of the elements that each MPI process receives

    for (int i = 0; i < num_procs; ++i) {
        recvcounts[i] = (i < rows_remainder) ? (rows_per_proc + 1) * A.cols() : rows_per_proc * A.cols();
        displacements[i] = (i < rows_remainder) ? i * (rows_per_proc + 1) * A.cols() : (i * rows_per_proc + rows_remainder) * A.cols();
    }
    cout << "recvcounts: " << recvcounts[0] << " " << recvcounts[1] << endl;
    cout << "displacements: " << displacements[0] << " " << displacements[1] << endl;

    Vec gathered_data = Vec::Zero(A.rows()*A.cols()); // This vector needs to store all the elements of the matrix A^T

    MPI_Gatherv(local_AT_byrows_flatten.data(), local_AT_byrows_flatten.size(), MPI_DOUBLE, gathered_data.data(), recvcounts.data(),
                displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {

        // Fill AT_byrows with the gathered data
        for (size_t i = 0; i < AT_byrows.rows(); ++i) {
            for (size_t j = 0; j < AT_byrows.cols(); ++j) {
                AT_byrows(i, j) = gathered_data[j*A.cols()+i];
            }
        }

        // Display the matrix A and its transpose
        std::cout << "Matrix A:\n" << A << endl << endl;
        std::cout << "Vector A^T:\n" << AT_byrows << endl << endl;

        // Check with Eigen transpose
        Mat check = A.transpose();
        if (rank == 0) {
            cout << "||A^T_Eigen - A^T_my||:\n" << (check-AT_byrows).norm() << endl << endl;
        }
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}