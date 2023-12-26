#include "rSVD.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <iostream>
#include <filesystem> // C++17 or later
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>

// The main function for running the tests

int main(int /*argc*/, char** argv) {
    std::cout << "test rSVD reduced" << std::endl;

    // Get the path to the directory where the executable is located
    std::filesystem::path exePath = std::filesystem::absolute(argv[0]);
    std::filesystem::path exeDir = exePath.parent_path();
    // Get the path to the root of the project
    std::filesystem::path root = exeDir.parent_path();

    // Input and output directories
    std::filesystem::path inputDir = root / "data" / "input";
    std::filesystem::path outputDir = root / "data" / "output" / "rSVD" / "my";

    // Create output directory if it doesn't exist
    if (!std::filesystem::exists(outputDir))
        std::filesystem::create_directory(outputDir);

    // Array to store file names
    std::vector<std::string> fileNames;

    // Iterate over files in the input directory
    for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
        if (entry.is_regular_file()) {
            fileNames.push_back(entry.path().filename().string());
        }
    }

    // Loop over file names
    for (auto& fileName : fileNames) {
        // Construct the full path for input file
        std::filesystem::path inputFilePath = inputDir / fileName;

        // Load matrix from the input file
        Eigen::SparseMatrix<double> sparseMatrix;
        
        Eigen::loadMarket(sparseMatrix, inputFilePath.string());
        Eigen::MatrixXd A = Eigen::MatrixXd(sparseMatrix);

        // start calculating the time
        auto start = std::chrono::high_resolution_clock::now();

        // Perform QR decomposition
        int m = A.rows();
        int n = A.cols();
        int k = 10; // numerical rank (we need an algorithm to find it) or target rank
        int p = 5; // oversampling parameter, usually it is set to 5 or 10
        int l = k + p;
        Mat A_copy = A;
        Mat U = Mat::Zero(A.rows(), l);
        Vec S = Vec::Zero(l);
        Mat V = Mat::Zero(l, A.cols());
        rSVD(A, U, S, V);

        // Record the end time
        auto end = std::chrono::high_resolution_clock::now();

        Mat diagonalMatrix = S.asDiagonal();
        Mat A_2(m, n);
        Mat mid(m, l);
        mid = U * diagonalMatrix;
        A_2 = mid * V.transpose();

        Mat diff = A_copy - A_2;
        double norm_of_difference = (diff).norm();

        // Calculate the duration
        std::chrono::duration<double> duration = end - start;
        // Print the duration in seconds
        std::cout << "Dataset: " << fileName << "\n";
        std::cout << "Size: " << A.rows() << ", " << A.cols() << "\n";
        std::cout << "Execution time: " << duration.count() << " seconds" << "\n";
        std::cout << "norm of diff : " << norm_of_difference << "\n";
        std::cout << "-------------------------\n" << std::endl;;

        size_t lastDotPos = fileName.find_last_of('.');

        // Check if a dot was found and extract the substring before it
        if (lastDotPos != std::string::npos) 
        {
            auto filenameWithoutExtension = fileName.substr(0, lastDotPos);
            fileName = filenameWithoutExtension;
        }
        // Construct the full paths for output files
        std::filesystem::path outputSFilePath = outputDir / (fileName + "_S.mtx");
        std::filesystem::path outputUFilePath = outputDir / (fileName + "_U.mtx");

        // Write Q and R matrices to output files
        Eigen::saveMarket(S, outputSFilePath.string());
        Eigen::saveMarket(U, outputUFilePath.string());
    }

    
    return 0;
}