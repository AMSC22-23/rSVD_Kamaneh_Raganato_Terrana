#ifndef CLASS_PCA_HPP
#define CLASS_PCA_HPP

#include "../../include/SVD_class.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

template<SVDMethod method>
class PCA : public SVD<method> {
public:
    using Mat = Eigen::MatrixXd;
    using Vec = Eigen::VectorXd;

    // Constructors
    PCA(const Mat& data, bool normalize = false)
        : SVD<method>(data), data_(data), normalize_(normalize) {
            
        initialize();
    }

    void initialize() {
        std::cout << "Initializing PCA..." << std::endl;
        assertDataValid();
        std::cout << "Data is valid." << std::endl;

        // Create a copy of the original data
        Mat centeredData = data_;

        // Center the data
        mean_ = centeredData.colwise().mean();
        centeredData.rowwise() -= mean_.transpose();
        std::cout << "Data centered." << std::endl;

        // Normalize the data if required
        if (normalize_) {
            stddev_ = (centeredData.array().square().colwise().sum() / (centeredData.rows() - 1)).sqrt();
            centeredData.array().rowwise() /= stddev_.transpose().array();
            std::cout << "Data normalized." << std::endl;
        }

        // Perform SVD on the centered (and normalized) data
        SVD<method>::setData(centeredData);
        SVD<method>::compute();
        std::cout << "SVD computation done." << std::endl;
    }

    // Method to ensure data is valid
    void assertDataValid() const {
        if (data_.rows() < 2 || data_.cols() < 2) {
            throw std::invalid_argument("PCA requires at least 2 rows and 2 columns.");
        }
    }

    // Data management
    void addData(const Mat& newData) {
        data_.conservativeResize(data_.rows() + newData.rows(), Eigen::NoChange);
        data_.bottomRows(newData.rows()) = newData;
        initialize();
    }

    void normalizeData() {
        stddev_ = (data_.array().square().colwise().sum() / (data_.rows() - 1)).sqrt();
        data_.array().rowwise() /= stddev_.transpose().array();
    }

    // Configuration options
    void setNormalization(bool normalize) {
        normalize_ = normalize;
        initialize();
    }

    // Variance methods
    Vec explainedVariance() const {
        Vec singular_values = SVD<method>::getS();
        return singular_values.array().square() / (data_.rows() - 1);
    }

    Vec explainedVarianceRatio() const {
        Vec variances = explainedVariance();
        return variances / variances.sum();
    }

    // Access to principal components and scores
    Mat principalDirections() const {
        return SVD<method>::getV();
    }

    Mat scores() const {
        return SVD<method>::getU() * SVD<method>::getS().asDiagonal();
    }

    // Projection and reconstruction
    Mat projectToPCA(const Mat& data) {
        return (data.rowwise() - mean_.transpose()) * SVD<method>::getV();
    }

    Mat reconstructFromPCA(const Mat& pcData) {
        return (pcData * SVD<method>::getV().transpose()).rowwise() + mean_.transpose();
    }

    void saveResults(const std::string& filename) {
        std::ofstream outFile(filename);

        // Save explained variance ratio
        Vec variance_ratio = explainedVarianceRatio();
        outFile << "Explained Variance Ratio:\n";
        for (int i = 0; i < variance_ratio.size(); ++i) {
            outFile << variance_ratio[i] << std::endl;
        }

        // Save scores (principal components)
        Mat scores = this->scores();
        outFile << "\nScores:\n";
        for (int i = 0; i < scores.rows(); ++i) {
            for (int j = 0; j < scores.cols(); ++j) {
                outFile << scores(i, j);
                if (j < scores.cols() - 1) outFile << ", ";
            }
            outFile << std::endl;
        }
        outFile.close();
    }

    // Check orthogonality of eigenvectors
    double checkOrthogonality() const {
        auto V = SVD<method>::getV();
        auto VtV = V.transpose() * V;
        return (VtV - Mat::Identity(V.cols(), V.cols())).norm();
    }

    void summary() const {
        Vec singular_values = (SVD<method>::getS())/std::sqrt(data_.rows()-1);  // Get singular values from SVD
        int numComponents = singular_values.size();
        double total_variance = (singular_values.array().square().sum())/(data_.rows()-1);  // Total variance is the sum of squared singular values

        Vec proportion_variance = ((singular_values.array().square() )/(data_.rows()-1))/ total_variance;
        Vec cumulative_variance = proportion_variance;

        // Calculate cumulative variance
        for (int i = 1; i < numComponents; ++i) {
            cumulative_variance[i] += cumulative_variance[i - 1];
        }

        // Print the results in a formatted manner
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Importance of components:\n";
        std::cout << "                           ";
        for (int i = 1; i <= numComponents; ++i) {
            std::cout << "Comp." << i << "       ";
        }
        std::cout << std::endl;

        // Standard deviations
        std::cout << "Standard deviation     ";
        for (int i = 0; i < numComponents; ++i) {
            std::cout << singular_values[i] << " ";
        }
        std::cout << std::endl;

        // Proportion of variance
        std::cout << "Proportion of Variance    ";
        for (int i = 0; i < numComponents; ++i) {
            std::cout << proportion_variance[i] << " ";
        }
        std::cout << std::endl;

        // Cumulative proportion of variance
        std::cout << "Cumulative Proportion     ";
        for (int i = 0; i < numComponents; ++i) {
            std::cout << cumulative_variance[i] << " ";
        }
        std::cout << std::endl;
    }

private:
    Mat data_;
    Vec mean_;
    Vec stddev_;
    bool normalize_;
};

#endif // CLASS_PCA_HPP