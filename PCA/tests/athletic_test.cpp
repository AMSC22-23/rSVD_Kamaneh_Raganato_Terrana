#include "class_pca.hpp"
#include <iostream>
#include <chrono>
Mat csv_to_mat(std::string & path){
  
    // Leggi il file CSV in una matrice Eigen
    std::vector<double> data;
    std::ifstream file(path);
    std::string line;
    int rows = 0;
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            data.push_back(std::stod(cell));
        }
        ++rows;
    }
    Eigen::MatrixXd A(rows, data.size()/rows);
int idx = 0;
for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < data.size()/rows; ++j) {
        A(i, j) = data[idx++];
    }
}
return A;
}
int main() {
    // Define the dataset file name
    //const std::string filename = "dataset_athletic.txt";
     std::string filename = "";

    // Load the dataset
    std::cout << "Loading dataset from " << filename << "...\n";
    Mat data = (csv_to_mat(filename)).transpose();
std::cout << "data size: " << data.rows() << " " << data.cols() << std::endl;

    // Create a PCA object
    bool use_svd = false;
    bool full_matrices = false;
    std::cout << "Creating PCA object...\n";
    PCA pca(data, use_svd, full_matrices);
    
    //SVD svd(data, full_matrices);
    //int target_rank=20;
    auto start = std::chrono::high_resolution_clock::now();
    pca.compute();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time taken by compute() using JacobiSVD: " << diff.count() << " s\n";
    
    return 0;
}