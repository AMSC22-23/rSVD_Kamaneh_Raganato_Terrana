#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <mpi.h>
#include <chrono>
#include <Eigen/Dense>
#include "rSVD.hpp"
#include "stb_image.h"
#include "stb_image_write.h"

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) std::cout << "main running" << std::endl;
    
    // const char* filename = "data/input/img/download.png";
    const char* filename = "data/input/img/Mona_Lisa-256x256(2).png";
    const char* outputFilename = "data/output/output_image.png";
    // Load image using stb_image
    int width, height, channels;
    stbi_uc* image = stbi_load(filename, &width, &height, &channels, 0);


    if (image == nullptr) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return -1;
    }

    // Convert image data to an Eigen MatrixXd
    Eigen::MatrixXd imageMatrix(height, width * channels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width * channels; ++x) {
            imageMatrix(y, x) = static_cast<double>(image[y * width *channels + x]);
        }
    }

    if (rank == 0)
    std::cout << "matrix size: " << imageMatrix.rows() << " x " << imageMatrix.cols() << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    int m = imageMatrix.rows(); 
    int n = imageMatrix.cols();
    int k = 5; // numerical rank (we need an algorithm to find it) or target rank
    int p = 20; // oversampling parameter, usually it is set to 5 or 10
    int l = k + p;
    Mat imageMatrix_copy = imageMatrix;
    Mat U = Mat::Zero(m, l);
    Vec S = Vec::Zero(l);
    Mat V = Mat::Zero(l, n);
    rSVD(imageMatrix, U, S, V, l);

    // Record the end time
    auto end = std::chrono::high_resolution_clock::now();

    Mat diagonalMatrix = S.asDiagonal();
    Mat A_2(m, n);
    Mat mid(m, l);
    mid = U * diagonalMatrix;
    A_2 = mid * V.transpose();

    Mat diff = imageMatrix_copy - A_2;
    double norm_of_difference = (diff).norm();
    std::chrono::duration<double> duration = end - start;

    if (rank == 0){
        // std::cout << "\nDataset: " << fileName << "\n";
        // std::cout << "Size: " << A.rows() << ", " << A.cols() << "\n";
        std::cout << "Number of Processors: " << num_procs << "\n";
        std::cout << "Execution time: " << duration.count() << " seconds" << "\n";
        std::cout << "norm of diff : " << norm_of_difference << "\n";
        std::cout << "-------------------------\n" << std::endl;
    }

    // Convert the Eigen matrix back to image data
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width * channels; ++x) {
            image[y * width * channels + x] = static_cast<stbi_uc>(A_2(y, x));
        }
    }

    stbi_write_png(outputFilename, width, height, channels, image, width * channels);

    stbi_image_free(image);
    MPI_Finalize();
    return 0;
}
