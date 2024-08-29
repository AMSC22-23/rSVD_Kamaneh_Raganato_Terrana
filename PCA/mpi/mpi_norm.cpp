// COMPUTE THE EUCLIDEAN NORM OF A VECTOR

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

    // Define the size of the vector
    const int size = 17;

    // Define a vector b and initialize with random values
    Vec b = Vec::Random(size);
    double norm = 0.0;

    // Divide b among processors
    int elem_per_proc = b.size() / num_procs;
    int elem_remainder = b.size() % num_procs;
    int local_elem = (rank < elem_remainder) ? elem_per_proc + 1 : elem_per_proc;
    int offset = rank * elem_per_proc + std::min(rank, elem_remainder);

    Vec local_b = Vec::Zero(local_elem);
    for (size_t i = 0; i < local_elem; ++i) {
        local_b[i] = b[offset+i];
    }
    Vec local_norm = Vec::Zero(1);

    // Compute a partial sum of the squares of the elements of local_b
    for (size_t i = 0; i < local_elem; ++i) {
        local_norm[0] += local_b[i] * local_b[i];
    }

    // Gather local results using MPI_Gatherv
    // Compute displacements and recvcounts arrays
    std::vector<int> recvcounts(num_procs); // Number of elements that each MPI process receives
    std::vector<int> displacements(num_procs); // Starting index of the elements that each MPI process receives

    // Since each MPI process compute one partial sum of the squares, we can manually set recvcounts to 1 and displacements to i
    for (int i = 0; i < num_procs; ++i) {
        recvcounts[i] = 1;
        displacements[i] = i;
    }
    // cout << "recvcounts: " << recvcounts[0] << " " << recvcounts[1] << endl;
    // cout << "displacements: " << displacements[0] << " " << displacements[1] << endl;

    Vec gathered_data = Vec::Zero(num_procs); // This vector needs to store all the elements of the vector b

    MPI_Gatherv(local_norm.data(), local_norm.size(), MPI_DOUBLE, gathered_data.data(), recvcounts.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {

        // Compute the total sum of the squares with the gathered data
        for (size_t i = 0; i < num_procs; ++i) {
            norm += gathered_data[i];
        }

        // Compute the square root
        norm = sqrt(norm);

        // Check with Eigen norm
        if (rank == 0) {
            cout << "Norm of b computed with Eigen: " << b.norm() << endl;
            cout << "Norm of b computed with MPI:   " << norm << endl;
        }
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}