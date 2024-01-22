#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "image_comp.hpp"
#include "stb_image.h"
#include "stb_image_write.h"

// Constructors
Image::Image() {

}

Image::Image(int width, int height) {
    // Implementation
}

void Image::load(const char *filename) {
    unsigned char* stb_img = stbi_load(filename, &originalWidth, &originalHeight, &channels, 1);

    if (stb_img == nullptr) {
        std::cerr << "Failed to load image: " << filename << std::endl;
    }
    // later on change to private member image_matrix 
    // instead of original matrix
    Eigen::MatrixXd tempMatrix(originalHeight, originalWidth);
    for (int i = 0; i < originalHeight; ++i) {
        for (int j = 0; j < originalWidth; ++j) {
            tempMatrix(i, j) = static_cast<double>(stb_img[i * originalWidth + j]);
        }
    }
    image_matrix = tempMatrix.transpose();
}

void Image::save(const char *filename) {
    // Convert Eigen matrix data to unsigned char array
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> image_data = image_matrix.cast<unsigned char>();
    int write_result = stbi_write_png(filename, originalWidth, originalHeight, 1, image_data.data(), originalWidth * sizeof(unsigned char));
    if (!write_result) {
        std::cerr << "Error saving image." << std::endl;
    }
}

void Image::save_compressed(const char *filename) {
    // saving 3 matrices into a binary file
    // saving the dimensions as metadata
    
    
}

void Image::decompress(const char *filename) {
    // read from a binary file and fill the 
    // 3 matrices

}

Eigen::MatrixXd Image::reconstruct() {
    // reconstruct the original matrix by multiplying
    // the 3 SVD matrices together

}


void Image::downscale(int scale_factor) {
    // Implementation
}

void Image::upscale(int scale_factor) {
    // Implementation
}

void Image::normalize() {
    // to make the operation more numerically stable
    original_min = image_matrix.minCoeff();
    original_max = image_matrix.maxCoeff();
    image_matrix = (image_matrix.array() - original_min) / (original_max - original_min);
}

void Image::deNormalize() {
    // to make the operation more numerically stable
    image_matrix = image_matrix.array() * (original_max - original_min) + original_min;
}

void Image::compress() {

    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (num_procs == 4){
    

    int m = image_matrix.rows();
    int n = image_matrix.cols();
    int k = 50; // numerical rank (we need an algorithm to find it) or target rank
    int p = 10; // oversampling parameter, usually it is set to 5 or 10
    int l = k + p;
    Eigen::MatrixXd imageMatrix_copy = image_matrix;
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(m, l);
    Eigen::VectorXd S = Eigen::VectorXd::Zero(l);
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(l, n);
    rSVD(imageMatrix_copy, U, S, V, l);

    left_singular = U;
    singular = S.asDiagonal();
    right_singular = V;

    // remove this later
    Eigen::MatrixXd temp = left_singular * singular;
    image_matrix = temp * right_singular.transpose();

    }
}
