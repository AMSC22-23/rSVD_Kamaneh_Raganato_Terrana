#ifndef CLASS_PCA_HPP
#define CLASS_PCA_HPP

#include "SVD_class.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>

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
        assertDataValid();
        centerData();
        if (normalize_) {
            normalizeData();
        }
        this->compute();
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

    // Normalization and centering
    void centerData() {
        mean_ = data_.colwise().mean();
        data_.rowwise() -= mean_.transpose();
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

    void setSolver(SVDMethod newMethod) {
        // Assuming your SVD class can handle this
        this->method = newMethod;  // Adjust SVD class design as necessary
        initialize();
    }

    void bootstrap(int numSamples) {
        // Placeholder for bootstrapping functionality
    }

    // Variance methods
    Vec explainedVariance() const {
        Vec singular_values = this->getS();
        return singular_values.array().square() / (data_.rows() - 1);
    }

    Vec explainedVarianceRatio() const {
        Vec variances = explainedVariance();
        return variances / variances.sum();
    }

    // Access to principal components and scores
    Mat principalComponents() const {
        return this->getV();
    }

    Mat scores() const {
        return this->getU() * this->getS().asDiagonal();
    }

    // Projection and reconstruction
    Mat projectToPCA(const Mat& data) {
        return (data.rowwise() - mean_.transpose()) * this->getV();
    }

    Mat reconstructFromPCA(const Mat& pcData) {
        return (pcData * this->getV().transpose()).rowwise() + mean_.transpose();
    }

    // Saving and loading state
   /* void save(const std::string& filename) const {
        std::ofstream file(filename);
        file << data_;
        file << mean_;
        file << stddev_;
    }

    void load(const std::string& filename) {
        std::ifstream file(filename);
        file >> data_;
        file >> mean_;
        file >> stddev_;
        initialize();
    }
*/
    // Check orthogonality of eigenvectors
    double checkOrthogonality() const {
        auto V = this->getV();
        auto VtV = V.transpose() * V;
        return (VtV - Mat::Identity(V.cols(), V.cols())).norm();
    }

private:
    Mat data_;
    Vec mean_;
    Vec stddev_;
    bool normalize_;
};

#endif // CLASS_PCA_HPP

