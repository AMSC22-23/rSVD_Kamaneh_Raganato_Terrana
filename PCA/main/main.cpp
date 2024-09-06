#include <iostream>
#include <string>
#include "PCA_class.hpp"

Mat_m loadDataset(const std::string& filename) {
        std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    std::vector<double> data;
    std::string line;
    int rows = 0;
    while (std::getline(file, line)) {
        if (rows == 0) { // Skip header row
            ++rows;
            continue;
        }
        std::stringstream lineStream(line);
        std::string cell;
        int cols=0;
        while (std::getline(lineStream, cell, ' ')) {
            if(cols==0){
                cols++;
                continue;
            }
            if (!cell.empty() && cell != "\"\"") {
                data.push_back(std::stod(cell));
            }
        }
        ++rows;
    }

    int cols = data.size() / (rows - 1); // Subtract 1 because we skipped the header row
    Mat_m matrix(rows - 1, cols);
    for (int i = 0; i < rows - 1; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = data[i * cols + j];
        }
    }

    return matrix;
    }


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_dataset> <normalize: yes/no>" << std::endl;
        return 1;
    }

    std::string dataset_path = argv[1];
    bool normalize = (std::string(argv[2]) == "yes");

    std::cout << "Running PCA on the dataset: " << dataset_path;
    std::cout << (normalize ? " with normalization." : " without normalization.") << std::endl;

    Mat_m data = loadDataset(dataset_path);
    PCA<SVDMethod::ParallelJacobi> pca(data, normalize);

    pca.summary();

    return 0;
}