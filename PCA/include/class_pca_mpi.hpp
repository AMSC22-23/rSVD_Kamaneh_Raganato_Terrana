#ifndef CLASS_PCA_HPP
#define CLASS_PCA_HPP

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <stdexcept>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <mpi.h>

#include "SVD_class.hpp"

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

class PCA : public SVD {
public:
    PCA(Eigen::MatrixXd& data, bool use_svd , bool full_matrices ,int rank,int size) 
        : data_(data), use_svd_(use_svd), SVD(data, full_matrices),mpi_rank_(rank),mpi_size_(size){}

    void compute() override;
    void setEigenvalues(const Vec& values);
    void setEigenvectors(const Mat& vectors);
    virtual Eigen::MatrixXd getV() const override;

    virtual  Eigen::VectorXd getS() const override;
    
    Mat scores();
    
    Mat computeCovarianceMatrix();
    static Mat loadDataset(const std::string& filename);
    void mean_center(Mat& data);
    Vec explained_variance() const;
    Vec explained_variance_ratio() const;
    Mat get_loadings() const;

    

private:
    Mat U_; // Matrice U ottenuta dalla SVD
    Vec S_; // Vettore dei valori singolari ottenuto dalla SVD
    Mat V_; // Matrice V ottenuta dalla SVD

    Mat eigenvectors; // Vettori propri ottenuti dalla decomposizione degli autovalori
    Vec eigenvalues; // Autovalori ottenuti dalla decomposizione degli autovalori

    Mat data_; // I dati originali

    bool use_svd_;
    std::string filename_;
    Mat scores_;

    int mpi_rank_; // MPI rank
    int mpi_size_; // MPI size
};

#endif // CLASS_PCA_HPP