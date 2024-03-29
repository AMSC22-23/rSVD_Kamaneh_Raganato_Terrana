#include "QR.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <mpi.h>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <filesystem> // C++17 or later
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

// The main function for running the tests
int main(int argc, char** argv) {
    using namespace std;
    using namespace Eigen;

    using Mat = MatrixXd;
    using Vec = VectorXd;

    MPI_Init(&argc, &argv);
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank==0) std::cout << "*** QR test 1 ***\n" << std::endl;
    // Get the path to the directory where the executable is located
    std::filesystem::path exePath = std::filesystem::absolute(argv[0]);
    std::filesystem::path exeDir = exePath.parent_path();
    // Get the path to the root of the project
    std::filesystem::path root = exeDir.parent_path();

    // Input and output directories
    std::filesystem::path inputDir = root / "data" / "input" / "mat";
    std::filesystem::path outputDir = root / "data" / "output" / "QR";

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
        Mat Q = Mat::Identity(m, std::min(m, n));
        Mat R = A.topRows(std::min(m, n));
        QRReducedDecomposition<double> qrReduced(A);
        qrReduced.decompose(Q, R);

        // Record the end time
        auto end = std::chrono::high_resolution_clock::now();

        Mat A_2(m, n);
        A_2 = Q * R;
        Mat diff = A - A_2;
        double norm_of_difference = (A - A_2).norm();

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
        std::filesystem::path outputQFilePath = outputDir / (fileName + "_Q.mtx");
        std::filesystem::path outputRFilePath = outputDir / (fileName + "_R.mtx");

        // Write Q and R matrices to output files
        Eigen::saveMarket(Q, outputQFilePath.string());
        Eigen::saveMarket(R, outputRFilePath.string());
    }

    MPI_Finalize();
    return 0;
}
