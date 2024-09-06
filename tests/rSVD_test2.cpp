#include <iostream>
#include <Eigen/Dense>

#include <queue>
#include <omp.h>

#include <execution>

#include <cmath>
#include <limits>
#include <Eigen/QR>
#include <iostream>
#include <chrono>

#include <iostream>
#include <cmath>
#include <fstream>
#include <ctime>
#include <string>
#include <sstream>
#include <mpi.h>
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <algorithm>
#include "Jacobi_Class.hpp"
#include "JacobiOperations.hpp"
#include "SVD_class.hpp"
#include "QR.hpp"
#include "rSVD.hpp"
#include "PM.hpp"

using Mat=Eigen::MatrixXd;
using Vec=Eigen::VectorXd;



double calculatePrecision(const Mat &A, const Mat &U, const Vec &Sigma, const Mat &V) {
    // Ricostruzione della matrice originale da U, Sigma, V
    Mat SigmaMat = Sigma.asDiagonal();
    Mat A_reconstructed = U * SigmaMat * V.transpose();
    
    // Calcolo dell'errore relativo
    double norm_diff = (A - A_reconstructed).norm();
    double norm_A = A.norm();
    return norm_diff / norm_A;
}
double calculatePrecisionPM(const Mat &A, const Mat &U, Mat &Sigma, const Mat &V) {
    // Ricostruzione della matrice originale da U, Sigma, V
    Mat A_reconstructed = U * Sigma * V.transpose();
    
    // Calcolo dell'errore relativo
    double norm_diff = (A - A_reconstructed).norm();
    double norm_A = A.norm();
    return norm_diff / norm_A;
}

void exportTimingAndPrecisionData(const std::string &filename, const std::vector<std::tuple<int, double, double, double, double, double, double>> &data) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "Rank,TimeJacobi(ms),TimePower(ms),TimeDynamicJacobi(ms),PrecisionJacobi,PrecisionPower,PrecisionDynamicJacobi\n";
        for (const auto &entry : data) {
            file << std::get<0>(entry) << "," << std::get<1>(entry) << "," << std::get<2>(entry) << "," 
                 << std::get<3>(entry) << "," << std::get<4>(entry) << "," << std::get<5>(entry) << "," 
                 << std::get<6>(entry) << "\n";
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv); // Initialize MPI
    

    // Vector to collect timing and precision data
    std::vector<std::tuple<int, double, double, double, double, double, double>> data;

    // Matrix dimensions to test
    int m = 250, n = 250;
    std::vector<int> ranks = {10, 20, 50, 70, 100, 120, 150, 170, 200, 250};  // You can add more ranks
    Mat A = Eigen::MatrixXd::Random(m, n);
    Mat U(m, n), V(n, n);
    Vec Sigma(n);

    Mat U1(m, n), V1(n, n);
    Vec Sigma1(n);

    Mat U2(m, n), V2(n, n);
    Vec Sigma2(n);

    for (int rank : ranks) {
        // Measure time for rSVD with Jacobi method
        std::cout<<"check1"<<std::endl;
        auto start_jacobi = std::chrono::high_resolution_clock::now();
        rSVD(A, U, Sigma, V, rank, SVDMethod::Jacobi);
        auto end_jacobi = std::chrono::high_resolution_clock::now();
        double timeJacobi = std::chrono::duration_cast<std::chrono::milliseconds>(end_jacobi - start_jacobi).count();

        // Calculate precision for rSVD with Jacobi method
        double precisionJacobi = calculatePrecision(A, U, Sigma, V);
        std::cout<<"check2"<<std::endl;
        // Measure time for rSVD with Power method
        auto start_power = std::chrono::high_resolution_clock::now();
        rSVD(A, U1, Sigma1, V1, rank, SVDMethod::Power);
        std::cout<<"rsvd power finished"<<std::endl;
        auto end_power = std::chrono::high_resolution_clock::now();
        double timePower = std::chrono::duration_cast<std::chrono::milliseconds>(end_power - start_power).count();

        // Calculate precision for rSVD with Power method
        std::cout<<"U1 dim: "<<U1.rows()<<" x "<<U1.cols()<<std::endl;
        std::cout<<"Sigma1 dim: "<<Sigma1.size()<<std::endl;
        std::cout<<"V1 dim: "<<V1.rows()<<" x "<<V1.cols()<<std::endl;
        // Trasforma Sigma1 in una matrice diagonale
        Mat Sigma1_mat = Mat::Zero(rank, A.cols());
        for (int i = 0; i < rank; ++i) {
            Sigma1_mat(i, i) = Sigma1[i];
        }

        double precisionPower = calculatePrecisionPM(A, U1, Sigma1_mat, V1);
        std::cout<<"check3"<<std::endl;
        // Measure time for rSVD with DynamicJacobi method
        auto start_dynamic = std::chrono::high_resolution_clock::now();
        rSVD(A, U2, Sigma2, V2, rank, SVDMethod::ParallelJacobi);
        auto end_dynamic = std::chrono::high_resolution_clock::now();
        double timeDynamicJacobi = std::chrono::duration_cast<std::chrono::milliseconds>(end_dynamic - start_dynamic).count();

        // Calculate precision for rSVD with DynamicJacobi method
        double precisionDynamicJacobi = calculatePrecision(A, U2, Sigma2, V2);

        // Add results to the data vector
        data.emplace_back(rank, timeJacobi, timePower, timeDynamicJacobi, precisionJacobi, precisionPower, precisionDynamicJacobi);
    }

    // Export results to a CSV file
    exportTimingAndPrecisionData("rsvd_timing_and_precision_results2.csv", data);

   MPI_Finalize(); // Finalizza MPI
    return 0;
}