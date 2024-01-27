#include <iostream>
#include <mpi.h>
#include <chrono>
#include <Eigen/Dense>
#include <cstring>
#include "rSVD.hpp"
#include "image_comp.hpp"


int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Check if the user provided a command line argument
    
    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: %s <image name>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Get the string from the command line argument
    const std::string user_string = argv[1];

    // Get the path to load and save image
    const std::string base_string_input = "data/input/img/";
    const std::string filename_input = "data/input/img/" + user_string;
    const std::string filename_output = "data/output/img/" + user_string;
    const std::string filename_output_compressed = "data/output/img/" + user_string + ".dat";
    

    // Record the start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Load, process, and save the image in parallel
    if(filename_input != base_string_input){

        // Create the image instance
        Image myImage;

        // Load image from file
        myImage.load(filename_input);

        // Downscale the image by a factor of 2
        myImage.downscale(1);

        // Normalize pixel values
        myImage.normalize();

        // Perform parallel compression using rSVD
        myImage.compress_parallel(60);

        // De-normalize pixel values
        myImage.deNormalize();

        // Upscale the image
        myImage.upscale(1);

        // Save the processed image and its compressed version
        if (rank == 0) 
        {
            myImage.save(filename_output);
            myImage.save_compressed(filename_output_compressed);
        }

        // Record the end time
        auto end_time = std::chrono::high_resolution_clock::now();

        // Calculate the duration
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        // Print the execution time
        if (rank == 0) {
            std::cout << "Compressed image saved as : " << filename_output_compressed << "\n";
            std::cout << "Compression ration : " << myImage.get_compression_ratio() << "\n";
            std::cout << "Execution time: " << duration << " milliseconds." << std::endl;
        }
        
    }


    MPI_Finalize();
    return 0;
}
