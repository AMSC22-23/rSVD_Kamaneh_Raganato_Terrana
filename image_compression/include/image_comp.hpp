#ifndef IMAGE_COMP_HPP
#define IMAGE_COMP_HPP

#include <iostream>
#include <mpi.h>
#include <chrono>
#include <Eigen/Dense>
#include "rSVD.hpp"

class Image {
private:
    // cv::Mat image; // If you decide to use cv::Mat, include the necessary OpenCV headers
    Eigen::MatrixXd image_matrix;
    int originalWidth;
    int originalHeight; 
    int channels;

public:
    // Constructor
    Image();
    Image(int width, int height);

    // Load image from file
    void load(const char *filename);

    // Save image to file
    void save(const char *filename);

    // Downscale the image
    void downscale(int scale_factor);

    // Upscale the image
    void upscale(int scale_factor);

    // Normalize pixel values to the range [0, 1]
    void normalize();

    // Compress the image (dummy function for illustration)
    void compress();
};

#endif // IMAGE_COMP_HPP
