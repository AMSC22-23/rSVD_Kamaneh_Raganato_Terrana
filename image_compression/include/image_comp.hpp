#ifndef IMAGE_COMP_HPP
#define IMAGE_COMP_HPP

#include <iostream>
#include <fstream>
#include <mpi.h>
#include <chrono>
#include <Eigen/Dense>
#include "rSVD.hpp"

/**
 * @brief Class representing an image with compression capabilities.
 */


class Image {
private:
    Eigen::MatrixXd image_matrix; /**< Matrix representing the image. */

    Eigen::MatrixXd left_singular; /**< Matrix storing the left singular vectors during compression. */
    Eigen::VectorXd singular; /**< Vector storing the singular values during compression. */
    Eigen::MatrixXd right_singular; /**< Matrix storing the right singular vectors during compression. */

    int originalWidth; /**< Original width of the image. */
    int originalHeight; /**< Original height of the image. */
    int channels; /**< Number of color channels in the image. */

    double original_min; /**< Minimum pixel value in the original image. */
    double original_max; /**< Maximum pixel value in the original image. */

public:
    /**
     * @brief Default constructor for the Image class.
     */
    Image();

    /**
     * @brief Parameterized constructor for the Image class.
     * @param width The width of the image.
     * @param height The height of the image.
     */
    Image(int width, int height);

    /**
     * @brief Load image data from a file.
     * @param filename The name of the file from which to load the image.
     */
    void load(const char *filename);

    /**
     * @brief Save the image data to a file.
     * @param filename The name of the file to which the image will be saved.
     */
    void save(const char *filename);

    /**
     * @brief Save the compressed image data to a file.
     * @param filename The name of the file to which the compressed image will be saved.
     */
    void save_compressed(const char *filename);

    /**
     * @brief Load compressed image data from a file.
     * @param filename The name of the file from which to load the compressed image.
     */
    void load_compressed(const char *filename);

    /**
     * @brief Reconstruct the image from compressed data.
     * @return The reconstructed image matrix.
     */
    Eigen::MatrixXd reconstruct();

    /**
     * @brief Downscale the image by a specified scale factor.
     * @param scale_factor The factor by which to downscale the image.
     */
    void downscale(int scale_factor);

    /**
     * @brief Upscale the image by a specified scale factor.
     * @param scale_factor The factor by which to upscale the image.
     */
    void upscale(int scale_factor);

    /**
     * @brief Normalize pixel values to the range [0, 1].
     */
    void normalize();

    /**
     * @brief Denormalize pixel values to the original range.
     */
    void deNormalize();

    /**
     * @brief Compress the image using singular value decomposition.
     */
    void compress(int k = -1);

    /**
     * @brief Compress the image in parallel using MPI.
     */
    void compress_parallel(int k = -1);
};

#endif // IMAGE_COMP_HPP
