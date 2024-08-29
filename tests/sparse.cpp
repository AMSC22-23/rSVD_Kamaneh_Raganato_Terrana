#include <iostream>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv)
{
    int n = 100;
    if (argc != 0) {
        n = *argv[0];
    }

    // Define matrix
    SparseMatrix<double> mat(n,n);                           
    for (int i=0; i<n; i++) {
        mat.coeffRef(i, i) = 2.0;
	    if(i>0) mat.coeffRef(i, i-1) = -1.0;
        if(i<n-1) mat.coeffRef(i, i+1) = -1.0;	
    }

    // Export matrix
    std::string matrixFileOut("./Asparse.mtx");
    Eigen::saveMarket(mat, matrixFileOut);

    return 0;    
}
