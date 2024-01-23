#include <iostream>
#include <mpi.h>
#include <chrono>
#include <Eigen/Dense>
#include "rSVD.hpp"
#include "image_comp.hpp"


int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) std::cout << "main running" << std::endl;
    Image myImage;


    const char* filename = "data/input/img/Mona_Lisa-256x256(2).png";
    myImage.load(filename);
    // myImage.normalize();
    // // const char* filename = "data/input/img/Mona_Lisa-256x256(2).png";
    const char* outputFilename = "data/output/apple_compressed.png";
    myImage.compress_parallel();
    // myImage.deNormalize();
    if (rank == 0) myImage.save(outputFilename);
    // myImage.save_compressed("data/output/compressed_matrices.dat");

    MPI_Finalize();
    return 0;
}
