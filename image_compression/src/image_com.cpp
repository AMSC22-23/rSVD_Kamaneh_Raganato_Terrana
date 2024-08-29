#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "image_comp.hpp"
#include "stb_image.h"
#include "stb_image_write.h"

// Constructors
Image::Image() {

}

Image::Image(int width, int height) {
   originalWidth = width;
   originalHeight = height;
}

void Image::load(const std::string &filename) {
    // Use STB library to load image data
    const char* filename_cstr = filename.c_str();
    unsigned char* stb_img = stbi_load(filename_cstr, &originalWidth, &originalHeight, &channels, 1);

    // Check if image loading was successful
    if (stb_img == nullptr) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return; // Return early if loading fails
    }

    // Create a temporary matrix to hold the image data
    Eigen::MatrixXd tempMatrix(originalHeight, originalWidth);

    // Copy image data from STB format to Eigen matrix
    for (int i = 0; i < originalHeight; ++i) {
        for (int j = 0; j < originalWidth; ++j) {
            // Convert image data to double and store in the matrix
            tempMatrix(i, j) = static_cast<double>(stb_img[i * originalWidth + j]);
        }
    }
    // Transpose the matrix to match Eigen's default storage order
    image_matrix = tempMatrix.transpose();

    // Free memory allocated by stbi_load
    stbi_image_free(stb_img);
}

void Image::save(const std::string &filename) {
    // Convert Eigen matrix data to unsigned char array
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> image_data = image_matrix.cast<unsigned char>();
    
    // Use STB image write library to save image data as PNG
    const char* filename_cstr = filename.c_str();
    int write_result = stbi_write_png(filename_cstr, image_matrix.rows(), image_matrix.cols(), 1, image_data.data(), image_matrix.rows() * sizeof(unsigned char));
    
    // Check if saving was successful
    if (!write_result) {
        std::cerr << "Error saving image." << std::endl;
    }
}


/**
 * @brief Save compressed image data to a binary file.
 * 
 * @param filename The name of the file to which the compressed image will be saved.
 */
void Image::save_compressed(const std::string &filename) {
    // saving 3 matrices into a binary file
    // saving the dimensions as metadata

    // Open the binary file for writing
    std::ofstream file(filename, std::ios::binary);

    // Check if the file is open
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Save sizes of matrices as metadata
    int rows_U = left_singular.rows();
    int cols_U = left_singular.cols();
    int size_S = singular.size();
    int rows_V = right_singular.rows();
    int cols_V = right_singular.cols();

    // Write metadata to the file
    file.write(reinterpret_cast<const char*>(std::addressof(rows_U)), sizeof(int));
    file.write(reinterpret_cast<const char*>(std::addressof(cols_U)), sizeof(int));
    file.write(reinterpret_cast<const char*>(std::addressof(size_S)), sizeof(int));
    file.write(reinterpret_cast<const char*>(std::addressof(rows_V)), sizeof(int));
    file.write(reinterpret_cast<const char*>(std::addressof(cols_V)), sizeof(int));


    // Save matrices as bytes
    for (int i = 0; i < rows_U; ++i) {
        for (int j = 0; j < cols_U; ++j) {
            // Convert matrix elements to integer and save as bytes
            int value = static_cast<int>(left_singular(i, j));
            char byteValue = static_cast<char>(value & 0xFF);
            file.write(&byteValue, sizeof(char));
        }
    }

    for (int i = 0; i < size_S; ++i) {
        // Convert vector elements to integer and save as bytes
        int value = static_cast<int>(singular(i));
        char byteValue = static_cast<char>(value & 0xFF);
        file.write(&byteValue, sizeof(char));
    }

    for (int i = 0; i < rows_V; ++i) {
        for (int j = 0; j < cols_V; ++j) {
            // Convert matrix elements to integer and save as bytes
            int value = static_cast<int>(right_singular(i, j));
            char byteValue = static_cast<char>(value & 0xFF);
            file.write(&byteValue, sizeof(char));
        }
    }

    // Close the file
    file.close();
    
}


/**
 * @brief Load compressed image data from a binary file.
 *
 * @param filename The name of the file from which the compressed image will be loaded.
 */
void Image::load_compressed(const std::string &filename) {
    // Open the binary file for reading
    std::ifstream file(filename, std::ios::binary);

    // Check if the file is open
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

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
            // Read a byte from the file and convert to double
            char byteValue;
            file.read(&byteValue, sizeof(char));
            left_singular(i, j) = static_cast<double>(static_cast<unsigned char>(byteValue));
        }
    }

    for (int i = 0; i < size_S; ++i) {
        // Read a byte from the file and convert to double
        char byteValue;
        file.read(&byteValue, sizeof(char));
        singular(i) = static_cast<double>(static_cast<unsigned char>(byteValue));
    }

    for (int i = 0; i < rows_V; ++i) {
        for (int j = 0; j < cols_V; ++j) {
            // Read a byte from the file and convert to double
            char byteValue;
            file.read(&byteValue, sizeof(char));
            right_singular(i, j) = static_cast<double>(static_cast<unsigned char>(byteValue));
        }
    }
 
    file.close();
}


Eigen::MatrixXd Image::reconstruct() {
    // reconstruct the original matrix by multiplying
    // the three SVD matrices together

    Eigen::MatrixXd reconstructed_matrix = left_singular * singular.asDiagonal() * right_singular.transpose();
    return reconstructed_matrix;
}


void Image::downscale(int scale_factor /* = -1*/) {
    if (scale_factor == -1) {
        scale_factor = 2; // Default scale factor
    }

    int new_width = originalWidth / scale_factor;
    int new_height = originalHeight / scale_factor;

    // Create a new matrix for the downscaled image
    Eigen::MatrixXd downscaled_matrix(new_height, new_width);

    // Iterate over the downscaled matrix and copy elements from the original matrix with stride
    for (int i = 0; i < new_height; ++i) {
        for (int j = 0; j < new_width; ++j) {
            downscaled_matrix(i, j) = image_matrix(i * scale_factor, j * scale_factor);
        }
    }

    // Update the image matrix
    image_matrix = downscaled_matrix;

    // Update the original dimensions
    originalWidth = new_width;
    originalHeight = new_height;
}

void Image::upscale(int scale_factor /* = -1*/) {
    if (scale_factor == -1) {
        scale_factor = 2; // Default scale factor
    }

    int new_width = originalWidth * scale_factor;
    int new_height = originalHeight * scale_factor;

    // Create a new matrix for the upscaled image
    Eigen::MatrixXd upscaled_matrix(new_height, new_width);

    // Iterate over the original matrix and repeat each value at regular intervals
    for (int i = 0; i < originalHeight; ++i) {
        for (int j = 0; j < originalWidth; ++j) {
            upscaled_matrix.block(i * scale_factor, j * scale_factor, scale_factor, scale_factor)
                .setConstant(image_matrix(i, j));
        }
    }

    // Update the image matrix
    image_matrix = upscaled_matrix;

    // Update the original dimensions
    originalWidth = new_width;
    originalHeight = new_height;
}


/**
 * @brief Normalize pixel values of the image matrix to the range [0, 1].
 *        This operation is performed to make the normalization more numerically stable.
 */
void Image::normalize() {
    // Find the minimum and maximum pixel values in the original image matrix
    original_min = image_matrix.minCoeff();
    original_max = image_matrix.maxCoeff();

    // Check if the range is not empty to avoid division by zero
    if (original_min < original_max) {
        // Normalize pixel values to the range [0, 1]
        image_matrix = (image_matrix.array() - original_min) / (original_max - original_min);
    } else {
        // Handle the case where the range is empty or undefined (original_min >= original_max)
        std::cerr << "Warning: Unable to normalize image. Empty or undefined pixel value range." << std::endl;
    }
}

/**
 * @brief Restore pixel values of the image matrix to their original range.
 *        This operation undoes the previous normalization.
 */
void Image::deNormalize() {
    // Check if original_min is less than original_max to avoid potential issues
    if (original_min < original_max) {
        // Restore pixel values to their original range
        image_matrix = image_matrix.array() * (original_max - original_min) + original_min;
    } else {
        // Handle the case where the range is empty or undefined (original_min >= original_max)
        std::cerr << "Warning: Unable to deNormalize image. Empty or undefined pixel value range." << std::endl;
    }
}

/**
 * @brief Compress the image matrix using randomized Singular Value Decomposition (rSVD).
 *        The function computes the rSVD and stores the result in the member variables.
 *        The image matrix is then reconstructed using the computed SVD matrices.
 *  
 * @param k The target rank for the rSVD.
 */
void Image::compress(int k /* = -1*/) {
    // Get the dimensions of the original image matrix
    int m = image_matrix.rows();
    int n = image_matrix.cols();

    if (k == -1) {
        k = std::min(m, n) / 4; // Set default rank to 1/4 the size of the matrix
    }

    int p = 10; // oversampling parameter, usually it is set to 5 or 10
    int l = k + p;

    // Copy the original image matrix for the rSVD computation
    Eigen::MatrixXd imageMatrix_copy = image_matrix;

    // Initialize matrices for rSVD result
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(m, l);
    Eigen::VectorXd S = Eigen::VectorXd::Zero(l);
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(l, n);

    // Perform rSVD
    rSVD(imageMatrix_copy, U, S, V, l);
    degree = l;

    // Store the rSVD matrices in member variables
    left_singular = U;
    singular = S;
    right_singular = V;

}

/**
 * @brief Compress the image matrix in parallel using rank-revealing Singular Value Decomposition (rSVD) with MPI.
 *        The function distributes the computation across multiple MPI processes, performs rSVD, and reconstructs the image matrix.
 *
 * @param k The target rank for the rSVD.
 */
void Image::compress_parallel(int k /* = -1*/ ) {

    int numProcesses, myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    // This function only works with a square number of processors
    // e.g. 1, 4, 9, 16
    double squareRoot = std::sqrt(numProcesses);
    if (squareRoot != std::floor(squareRoot)){
        std::cout << "Number of processors is not square!" << std::endl;
        return;
    }
    
    int m = image_matrix.rows();
    int n = image_matrix.cols();

    if (k == -1) {
        k = std::min(m, n) / 4; // Set default rank to 1/4 the size of the matrix
    }
    int p = 10; // oversampling parameter, usually it is set to 5 or 10
    int l = k + p;

    degree = l;

    // defining the size of blocks to be decomposed in parallel
    int blockSizeRows = m / std::sqrt(numProcesses);
    int blockSizeCols = n / std::sqrt(numProcesses);

    Eigen::MatrixXd copy = image_matrix;

    // Each processor takes over a block in the column-major order
    int sqrtNumProcesses = static_cast<int>(std::sqrt(numProcesses));
    Eigen::MatrixXd localMatrix = image_matrix.block(
                    (myRank / sqrtNumProcesses) * blockSizeRows,  
                    (myRank % sqrtNumProcesses) * blockSizeCols,                                      
                    blockSizeRows,
                    blockSizeCols
                    );


    // Do rSVD computation in parallel on local matrices
    Eigen::MatrixXd imageMatrix_copy = image_matrix;
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(m, l);
    Eigen::VectorXd S = Eigen::VectorXd::Zero(l);
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(l, n);
    rSVD(localMatrix, U, S, V, l);

    // Reconstruct the local matrix
    localMatrix = U * S.asDiagonal() * V.transpose();

    // Gather the local matrices on processor 0
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

}

double Image::get_compression_ratio(){
    double initial_size = originalHeight * originalWidth;
    double compressed_size =  degree * (originalWidth + originalHeight + 1);

    return initial_size / compressed_size;
}
