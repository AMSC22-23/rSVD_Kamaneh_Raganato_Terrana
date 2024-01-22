#ifndef IMAGE_COMP_HPP
#define IMAGE_COMP_HPP

#include <H5Cpp.h>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <chrono>
#include <Eigen/Dense>
#include "rSVD.hpp"

class Image {
private:
    // cv::Mat image; // If you decide to use cv::Mat, include the necessary OpenCV headers
    Eigen::MatrixXd image_matrix;

    Eigen::MatrixXd left_singular;
    Eigen::MatrixXd singular;
    Eigen::MatrixXd right_singular;

    int originalWidth;
    int originalHeight; 
    int channels;

    double original_min;
    double original_max;

public:
    // Constructor
    Image();
    Image(int width, int height);

    // Load image from file
    void load(const char *filename);

    // Save image to file
    void save(const char *filename);

    void save_compressed(const char *filename);

    void decompress(const char *filename);

    Eigen::MatrixXd reconstruct();

    // Downscale the image
    void downscale(int scale_factor);

    // Upscale the image
    void upscale(int scale_factor);

    // Normalize pixel values to the range [0, 1]
    void normalize();

    void deNormalize();

    // Compress the image (dummy function for illustration)
    void compress();
};

#endif // IMAGE_COMP_HPP
