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
    int originalWidth, originalHeight, channels;
    stbi_uc* originalImage = stbi_load(filename, &originalWidth, &originalHeight, &channels, 0);

    if (originalImage == nullptr) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return -1;
    }

    // Convert original image data to grayscale
    Eigen::MatrixXd imageMatrix(originalHeight, originalWidth);
    for (int y = 0; y < originalHeight; ++y) {
        for (int x = 0; x < originalWidth; ++x) {
            // Assuming the image is grayscale, take the average of RGB values
            int pixelIndex = (y * originalWidth + x) * channels;
            imageMatrix(y, x) = static_cast<double>((originalImage[pixelIndex] + originalImage[pixelIndex + 1] + originalImage[pixelIndex + 2]) / 3) / 255.0;
        }
    }

    // Normalize the entire matrix
    imageMatrix /= imageMatrix.maxCoeff();

    if (rank == 0)
        std::cout << "matrix size: " << imageMatrix.rows() << " x " << imageMatrix.cols() << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    int m = imageMatrix.rows();
    int n = imageMatrix.cols();
    int k = 5; // numerical rank (we need an algorithm to find it) or target rank
    int p = 15; // oversampling parameter, usually it is set to 5 or 10
    int l = k + p;
    Eigen::MatrixXd imageMatrix_copy = imageMatrix;
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(m, l);
    Eigen::VectorXd S = Eigen::VectorXd::Zero(l);
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(l, n);
    rSVD(imageMatrix, U, S, V, l);

    // Record the end time
    auto end = std::chrono::high_resolution_clock::now();

    Eigen::MatrixXd diagonalMatrix = S.asDiagonal();
    Eigen::MatrixXd A_2(m, n);
    Eigen::MatrixXd mid(m, l);
    mid = U * diagonalMatrix;
    A_2 = mid * V.transpose();

    Eigen::MatrixXd diff = imageMatrix_copy - A_2;
    double norm_of_difference = diff.norm();
    std::chrono::duration<double> duration = end - start;

    size_t originalSize = sizeof(double) * imageMatrix_copy.rows() * imageMatrix_copy.cols();
    size_t compressedSize = sizeof(double) * (U.size() + S.size() + V.size());

    double compressionRatio = static_cast<double>(originalSize) / compressedSize;

    if (rank == 0) {
        std::cout << "Original size: " << originalSize << " bytes" << std::endl;
        std::cout << "Compressed size: " << compressedSize << " bytes" << std::endl;
        std::cout << "Compression ratio: " << compressionRatio << std::endl;

        std::cout << "Number of Processors: " << num_procs << "\n";
        std::cout << "Execution time: " << duration.count() << " seconds" << "\n";
        std::cout << "norm of diff : " << norm_of_difference << "\n";
        std::cout << "-------------------------\n" << std::endl;
    }

    // Convert the Eigen matrix back to image data
    for (int y = 0; y < originalHeight; ++y) {
        for (int x = 0; x < originalWidth; ++x) {
            // Clip values to ensure they are within the valid range [0, 255]
            originalImage[(y * originalWidth + x) * channels] = static_cast<stbi_uc>(std::min(255.0, std::max(0.0, A_2(y, x) * 255.0)));
        }
    }

    stbi_write_png(outputFilename, originalWidth, originalHeight, channels, originalImage, originalWidth * channels);

    stbi_image_free(originalImage);
    MPI_Finalize();
    return 0;
}
