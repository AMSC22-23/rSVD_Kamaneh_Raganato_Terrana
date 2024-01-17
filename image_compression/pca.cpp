#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;

MatrixXd computePCA(const MatrixXd& A, int k) {
    // Center the data by subtracting the mean of each column
    VectorXd mean = A.colwise().mean();
    MatrixXd centered = A.rowwise() - mean.transpose();

    // Compute the covariance matrix
    MatrixXd covariance = (centered.adjoint() * centered) / double(A.rows() - 1);

    // Compute the eigen decomposition of the covariance matrix
    SelfAdjointEigenSolver<MatrixXd> eigensolver(covariance);
    
    if (eigensolver.info() != Success) {
        // Handle error
        std::cerr << "Eigen decomposition failed!" << std::endl;
        // You may want to throw an exception or return an error code
    }

    // Get the top k eigenvectors corresponding to the largest eigenvalues
    MatrixXd topKEigenvectors = eigensolver.eigenvectors().rightCols(k);

    // Project the centered data onto the top k eigenvectors to get the principal components
    MatrixXd principalComponents = centered * topKEigenvectors;

    return principalComponents;
}

int main() {
    // Example usage
    MatrixXd inputMatrix(3, 3); // Replace this with your actual data
    inputMatrix << 1, 2, 3,
                   4, 5, 6,
                   7, 8, 9;

    int k = 3; // Number of principal components

    MatrixXd result = computePCA(inputMatrix, k);

    std::cout << "Principal Components:\n" << result << std::endl;

    return 0;
}
