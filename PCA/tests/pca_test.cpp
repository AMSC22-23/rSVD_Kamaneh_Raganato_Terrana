#include "PCA_class.hpp"
#include "SVD_class.hpp"
#include <Eigen/Dense>
#include <iostream>



Mat_m loadTouristsData(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file");
    }

    std::string line;
    std::vector<std::vector<double>> data;

    // Read the header line and ignore
    std::getline(file, line);

    // Read the file line by line
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        std::string value;
        std::vector<double> row;

        // Skip the first three columns (categorical variables)
        std::getline(stream, value, ' '); // Skip first column
        std::getline(stream, value, ' '); // Skip second column
        std::getline(stream, value, ' '); // Skip third column

        // Read the remaining columns
        while (stream >> value) {
            try {
                row.push_back(std::stod(value));
            } catch (const std::invalid_argument&) {
                //std::cerr << "Skipping invalid value: " << value << " in row: " << line << std::endl;
            }
        }

        if (row.size() == 8) { // Ensure the row has the correct number of columns
            data.push_back(row);
        } 
    }

    file.close();

    // Convert the vector of vectors to Eigen::MatrixXd
    size_t rows = data.size();
    size_t cols = rows > 0 ? data[0].size() : 0;
    Mat_m matrix(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix(i, j) = data[i][j];
        }
    }

    return matrix;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> [normalize]" << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    bool normalize = (argc > 3) ? (std::string(argv[3]) == "yes") : false;

    MPI_Init(NULL, NULL);

    try {
        Mat_m data = loadTouristsData(inputFile);

        // Creazione dell'oggetto PCA
        PCA<SVDMethod::ParallelJacobi> pca(data, normalize);

        pca.summary();
        pca.saveResults(outputFile);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}

