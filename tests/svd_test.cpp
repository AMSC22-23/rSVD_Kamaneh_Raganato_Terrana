#include <iostream>
#include "../include/SVD_class.hpp"
#include <Eigen/Dense>

int main() {
    // Definisci la matrice di input
    Eigen::MatrixXd A(5, 5);
    A << 1, 2, 3, 4, 5,
         2, 6, 7, 8, 9,
         3, 7, 10, 11, 12,
         4, 8, 11, 13, 14,
         5, 9, 12, 14, 15;

    // Crea un'istanza della classe SVD con il metodo Jacobi
    SVD<SVDMethod::Jacobi> svd(A);

    // Calcola la SVD
    svd.compute();

    // Ottieni i risultati
    Eigen::MatrixXd U(5,5);
    U = svd.getU();
    Eigen::VectorXd S(5);
    S = svd.getS();
    Eigen::MatrixXd V(5,5);
    V = svd.getV();

    // Stampa i risultati
    std::cout << "Matrice U:\n" << U << std::endl;
    std::cout << "Vettore S:\n" << S << std::endl;
    std::cout << "Matrice V:\n" << V << std::endl;

    std::cout<<"reconstructed matrix: "<<U*S.asDiagonal()*V.transpose()<<std::endl;

    return 0;
}