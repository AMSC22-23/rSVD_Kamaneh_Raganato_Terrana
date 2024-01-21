#include <iostream>
#include <mpi.h>
#include <chrono>
#include <Eigen/Dense>
#include "rSVD.hpp"
#include "stb_image.h"
#include "stb_image_write.h"

class Image {
private:
    // cv::Mat image;

public:
    // Constructor
    Image(int width, int height) {}

    // Load image from file
    void load(const std::string& filename) {
        // image = cv::imread(filename, cv::IMREAD_COLOR);

        if (image.empty()) {
            std::cerr << "Error: Could not open or read the image file." << std::endl;
        }
    }

    // Downscale the image
    void downscale(int scale_factor) {
        if (!image.empty()) {
            cv::resize(image, image, cv::Size(), 1.0 / scale_factor, 1.0 / scale_factor);
        } else {
            std::cerr << "Error: Image not loaded." << std::endl;
        }
    }

    // Upscale the image
    void upscale(int scale_factor) {
        if (!image.empty()) {
            cv::resize(image, image, cv::Size(), scale_factor, scale_factor);
        } else {
            std::cerr << "Error: Image not loaded." << std::endl;
        }
    }

    // Normalize pixel values to the range [0, 1]
    void normalize() {
        if (!image.empty()) {
            image.convertTo(image, CV_32F, 1.0 / 255.0);
        } else {
            std::cerr << "Error: Image not loaded." << std::endl;
        }
    }

    // Compress the image (dummy function for illustration)
    void compress() {
        // Placeholder for compression logic
        if (!image.empty()) {
            std::cout << "Image compression applied." << std::endl;
        } else {
            std::cerr << "Error: Image not loaded." << std::endl;
        }
    }
};

