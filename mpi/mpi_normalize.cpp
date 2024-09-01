// NORMALIZA A VECTOR

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
    Vec norm = Vec::Zero(1);

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

    // Since each MPI processo compute one partial sum of the squares, we can manually set recvcounts to 1 and displacements to i
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
            norm[0] += gathered_data[i];
        }

        // Compute the square root
        norm[0] = sqrt(norm[0]);

        // Check with Eigen norm
        if (rank == 0) {
            cout << "Norm of b computed with Eigen: " << b.norm() << endl;
            cout << "Norm of b computed with MPI:   " << norm[0] << endl;
        }
    }

    // Once the norm is computed, the vector can be normalized
    // Divide b among processors
    int elem_per_proc2 = b.size() / num_procs;
    int elem_remainder2 = b.size() % num_procs;
    int local_elem2 = (rank < elem_remainder2) ? elem_per_proc2 + 1 : elem_per_proc2;
    int offset2 = rank * elem_per_proc2 + std::min(rank, elem_remainder2);

    Vec local_b2 = Vec::Zero(local_elem2);
    for (size_t i = 0; i < local_elem2; ++i) {
        local_b2[i] = b[offset2+i];
    }

    // Broadcast the computed norm to all processors
    MPI_Bcast(norm.data(), norm.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Normalize each element of local_b
    for (size_t i = 0; i < local_elem2; ++i) {
        local_b2[i] = local_b2[i] / norm[0];
    }

    // Gather local results using MPI_Gatherv
    // Compute displacements and recvcounts arrays
    std::vector<int> recvcounts2(num_procs); // Number of elements that each MPI process receives
    std::vector<int> displacements2(num_procs); // Starting index of the elements that each MPI process receives

    // Since each MPI process compute one partial sum of the squares, we can manually set recvcounts to 1 and displacements to i
    for (int i = 0; i < num_procs; ++i) {
        recvcounts2[i] = (i < elem_remainder2) ? (elem_per_proc2 + 1) : elem_per_proc2;
        displacements2[i] = (i < elem_remainder2) ? i * (elem_per_proc2 + 1) : (i * elem_per_proc2 + elem_remainder2);
    }
    cout << "recvcounts2: " << recvcounts2[0] << " " << recvcounts2[1] << endl;
    cout << "displacements2: " << displacements2[0] << " " << displacements2[1] << endl;

    Vec b_normalized = Vec::Zero(b.size()); // This vector needs to store all the elements of the vector b

    MPI_Gatherv(local_b2.data(), local_b2.size(), MPI_DOUBLE, b_normalized.data(), recvcounts2.data(), displacements2.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Check with Eigen norm
    if (rank == 0) {
        cout << "Norm of b_normalized computed with Eigen: " << b_normalized.norm() << endl;
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}