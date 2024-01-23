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
    int write_result = stbi_write_png(filename, image_matrix.rows(), image_matrix.cols(), 1, image_data.data(), image_matrix.rows() * sizeof(unsigned char));
    if (!write_result) {
        std::cerr << "Error saving image." << std::endl;
    }
}

void Image::save_compressed(const char *filename) {
    // saving 3 matrices into a binary file
    // saving the dimensions as metadata
    std::ofstream file(filename, std::ios::binary);

    // Save sizes as metadata
    int rows_U = left_singular.rows();
    int cols_U = left_singular.cols();
    int size_S = singular.size();
    int rows_V = right_singular.rows();
    int cols_V = right_singular.cols();

    file.write(reinterpret_cast<const char*>(&rows_U), sizeof(int));
    file.write(reinterpret_cast<const char*>(&cols_U), sizeof(int));
    file.write(reinterpret_cast<const char*>(&size_S), sizeof(int));
    file.write(reinterpret_cast<const char*>(&rows_V), sizeof(int));
    file.write(reinterpret_cast<const char*>(&cols_V), sizeof(int));


    // Save matrices as bytes
    for (int i = 0; i < rows_U; ++i) {
        for (int j = 0; j < cols_U; ++j) {
            int value = static_cast<int>(left_singular(i, j));
            char byteValue = static_cast<char>(value & 0xFF);
            file.write(&byteValue, sizeof(char));
        }
    }

    for (int i = 0; i < size_S; ++i) {
        int value = static_cast<int>(singular(i));
        char byteValue = static_cast<char>(value & 0xFF);
        file.write(&byteValue, sizeof(char));
    }

    for (int i = 0; i < rows_V; ++i) {
        for (int j = 0; j < cols_V; ++j) {
            int value = static_cast<int>(right_singular(i, j));
            char byteValue = static_cast<char>(value & 0xFF);
            file.write(&byteValue, sizeof(char));
        }
    }
    file.close();
    
}

void Image::load_compressed(const char *filename) {
    // Loading matrices and vector from a binary file
    std::ifstream file(filename, std::ios::binary);

    // Read sizes from metadata
    int rows_U, cols_U, size_S, rows_V, cols_V;
    file.read(reinterpret_cast<char*>(&rows_U), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols_U), sizeof(int));
    file.read(reinterpret_cast<char*>(&size_S), sizeof(int));
    file.read(reinterpret_cast<char*>(&rows_V), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols_V), sizeof(int));

    // Resize matrices and vector
    left_singular.resize(rows_U, cols_U);
    singular.resize(size_S);
    right_singular.resize(rows_V, cols_V);

    // Read matrices and vector from bytes
    for (int i = 0; i < rows_U; ++i) {
        for (int j = 0; j < cols_U; ++j) {
            char byteValue;
            file.read(&byteValue, sizeof(char));
            left_singular(i, j) = static_cast<double>(static_cast<unsigned char>(byteValue));
        }
    }

    for (int i = 0; i < size_S; ++i) {
        char byteValue;
        file.read(&byteValue, sizeof(char));
        singular(i) = static_cast<double>(static_cast<unsigned char>(byteValue));
    }

    for (int i = 0; i < rows_V; ++i) {
        for (int j = 0; j < cols_V; ++j) {
            char byteValue;
            file.read(&byteValue, sizeof(char));
            right_singular(i, j) = static_cast<double>(static_cast<unsigned char>(byteValue));
        }
    }
 
    file.close();
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
    
    int m = image_matrix.rows();
    int n = image_matrix.cols();
    int k = 10; // numerical rank (we need an algorithm to find it) or target rank
    int p = 10; // oversampling parameter, usually it is set to 5 or 10
    int l = k + p;
    Eigen::MatrixXd imageMatrix_copy = image_matrix;
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(m, l);
    Eigen::VectorXd S = Eigen::VectorXd::Zero(l);
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(l, n);
    rSVD(imageMatrix_copy, U, S, V, l);

    left_singular = U;
    singular = S;
    right_singular = V;

    // remove this later
    Eigen::MatrixXd temp = left_singular * singular.asDiagonal();
    image_matrix = temp * right_singular.transpose();

}


void Image::compress_parallel() {

    int numProcesses, myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    double squareRoot = std::sqrt(numProcesses);
    if (squareRoot != std::floor(squareRoot)){
        std::cout << "Number of processors is not square!" << std::endl;
        return;
    }
    
    int m = image_matrix.rows();
    int n = image_matrix.cols();
    int k = 20; // numerical rank (we need an algorithm to find it) or target rank
    int p = 10; // oversampling parameter, usually it is set to 5 or 10
    int l = k + p;

    int blockSizeRows = m / std::sqrt(numProcesses);
    int blockSizeCols = n / std::sqrt(numProcesses);

    Eigen::MatrixXd copy(m, n);
    copy = image_matrix;

    int sqrtNumProcesses = static_cast<int>(std::sqrt(numProcesses));
    Eigen::MatrixXd localMatrix = image_matrix.block(
                    (myRank / sqrtNumProcesses) * blockSizeRows,  
                    (myRank % sqrtNumProcesses) * blockSizeCols,                                      
                    blockSizeRows,
                    blockSizeCols
                    );


    // Do stuff
    // ...
    Eigen::MatrixXd imageMatrix_copy = image_matrix;
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(m, l);
    Eigen::VectorXd S = Eigen::VectorXd::Zero(l);
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(l, n);
    rSVD(localMatrix, U, S, V, l);

    localMatrix = U * S.asDiagonal() * V.transpose();

    // // Copy the gathered matrix to the global matrix (only on the root)
    MPI_Status status;
    if (myRank == 0) {
        image_matrix.setZero();
        int rank = 0;
        image_matrix.block(
                    (rank / sqrtNumProcesses) * blockSizeRows,  
                    (rank % sqrtNumProcesses) * blockSizeCols,                                      
                    blockSizeRows,
                    blockSizeCols) = localMatrix;
        for (int i=1; i<numProcesses; i++){
            MPI_Recv(localMatrix.data(), blockSizeRows * blockSizeCols, MPI_DOUBLE,
                        i, 0, MPI_COMM_WORLD, &status);

            image_matrix.block(
                        (i / sqrtNumProcesses) * blockSizeRows,  
                        (i % sqrtNumProcesses) * blockSizeCols,                                      
                        blockSizeRows,
                        blockSizeCols) = localMatrix;
        }
        
    }
    else {
        // Send localMatrix data to processor 0
        MPI_Send(localMatrix.data(), blockSizeRows * blockSizeCols, MPI_DOUBLE,
                 0, 0, MPI_COMM_WORLD);
    }

    // // remove this later
    Eigen::MatrixXd temp = left_singular * singular.asDiagonal();
    image_matrix = temp * right_singular.transpose();

}
