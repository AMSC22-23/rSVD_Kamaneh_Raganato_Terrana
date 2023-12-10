#include "../include/powerMethod/PM1_omp.hpp"
#include "../include/powerMethod/SVD_omp.hpp"

int main() {

    int m = 10;
    int n = 50;
    Mat A = Mat::Random(m, n); // Mat::Random(m, n) generate a random matrix with entries in [-1, 1]

    const int dim = (A.rows() < A.cols()) ? A.rows() : A.cols();
    Vec sigma = Vec::Zero(dim);
    Mat U = Mat::Zero(A.rows(), dim); // U m*n or m*m
    Mat V = Mat::Zero(dim, A.cols()); // V n*n or m*n

    SVD(A, sigma, U, V, dim);

    cout << "THIN SVD" << endl;
    cout << "Singular values: " << endl << sigma << endl;
    // cout << "U = " << endl << U << endl;
    // cout << "V = " << endl << V << endl;

    // Export the singular values as csv file
    ofstream filecsv("sigma.csv");
    if (filecsv.is_open()) {
        Eigen::IOFormat fmt(Eigen::FullPrecision, Eigen::DontAlignCols, "\t", "\n");
        filecsv << sigma.format(fmt);
        filecsv.close();
    }

    /*
    // Export A as csv file
    ofstream filecsv("Arandom.csv");
    if (filecsv.is_open()) {
        Eigen::IOFormat fmt(Eigen::FullPrecision, Eigen::DontAlignCols, "\t", "\n");
        filecsv << A.format(fmt);
        filecsv.close();
    }

    // Export B = A^T*A as mtx file in Matrix Market format
    SpMat BS = B.sparseView(); // sparseView() returns a sparse view of the dense matrix B, it does not convert B into a sparse matrix
    string matrixFileOut("./Brandom.mtx");
    saveMarket(BS, matrixFileOut);
    */

    return 0;
}
