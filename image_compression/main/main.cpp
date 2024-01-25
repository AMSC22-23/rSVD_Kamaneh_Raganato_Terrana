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
    const char* user_string = argv[1];

    // Get the path to load and save image
    const char* base_string_input = "data/input/img/";
    const char* base_string_output = "data/output/img/";
    

    // Calculate the size of the concatenated string
    size_t size_input = std::strlen(base_string_input) + std::strlen(user_string) + 1;
    size_t size_output = std::strlen(base_string_output) + std::strlen(user_string) + 1;
    size_t size_output_compressed = std::strlen(base_string_output) + std::strlen(user_string) + std::strlen(".dat") + 1;
    
    // Allocate memory for the concatenated string
    char* filename_input = new char[size_input];
    char* filename_output = new char[size_output];
    char* filename_output_compressed = new char[size_output_compressed];

    // Copy the base string into the result buffer
    std::strcpy(filename_input, base_string_input);
    std::strcpy(filename_output, base_string_output);
    std::strcpy(filename_output_compressed, base_string_output);

    // Concatenate the user string onto the end
    std::strcat(filename_input, user_string);
    std::strcat(filename_output, user_string);
    std::strcat(filename_output_compressed, user_string);
    std::strcat(filename_output_compressed, ".dat");

    // Record the start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Load, process, and save the image in parallel
    if(filename_input != base_string_input){

        // Create the image instance
        Image myImage;

        // Load image from file
        myImage.load(filename_input);

        // Downscale the image by a factor of 2
        myImage.downscale(2);

        // Normalize pixel values
        myImage.normalize();

        // Perform parallel compression using rSVD
        myImage.compress_parallel(80);

        // De-normalize pixel values
        myImage.deNormalize();

        // Upscale the image
        myImage.upscale(2);

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


    // Free allocated memory
    delete[] filename_input;
    delete[] filename_output;
    delete[] filename_output_compressed;

    MPI_Finalize();
    return 0;
}
