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

void Image::downscale(int scale_factor) {
    // Implementation
}

void Image::upscale(int scale_factor) {
    // Implementation
}

void Image::normalize() {
    // Implementation
}

void Image::compress() {
    // Implementation
}
