#include "../include/class_pca.hpp"
#include <iostream>

int main() {
    // Define the dataset file name
    const std::string filename = "dataset_athletic.txt";

    // Load the dataset
    std::cout << "Loading dataset from " << filename << "...\n";
    Mat data = PCA::loadDataset(filename);

    // Create a PCA object
    bool use_svd = true;
    bool full_matrices = false;
    std::cout << "Creating PCA object...\n";
    PCA pca(data, use_svd, full_matrices);
    
    /*SVD svd(data, full_matrices);
    int target_rank=5;
    svd.rSVD(target_rank);
    std::cout <<"RSVD" <<svd.getS() << std::endl;*/


    // Print the data before computation
    //std::cout << "Data before PCA computation:\n" << data << "\n";

    // Perform PCA computation
    std::cout << "Computing PCA...\n";
    pca.compute();

    // Print the data after computation
    //std::cout << "Data after PCA computation:\n" << data << "\n";

    // Get and print the singular values
    Mat scores= pca.scores();
    std::cout << "principal components: \n" << scores << "\n";

    // Calculate and print the covariance matrix
    Mat covariance_matrix = pca.computeCovarianceMatrix();
    std::cout << "Covariance matrix:\n" << covariance_matrix << "\n";

    // Calculate and print the explained variance
    Vec explained_variance = pca.explained_variance();
    std::cout << "Explained variance:\n" << explained_variance << "\n";

    // Calculate and print the explained variance ratio
    Vec explained_variance_ratio = pca.explained_variance_ratio();
    std::cout << "Explained variance ratio:\n" << explained_variance_ratio << "\n";

    return 0;
}
