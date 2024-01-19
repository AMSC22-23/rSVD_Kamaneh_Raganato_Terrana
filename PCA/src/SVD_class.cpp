#include "SVD_class.hpp"
//#include <Eigen/SVD>



/*SVD::SVD(const Eigen::MatrixXd& data, bool full_matrices)
    : my_data(data), full_matrices_(full_matrices) {}*/

Eigen::MatrixXd SVD::getU() const {
    return U;
}

Eigen::VectorXd SVD::getS() const {
    return S;
}

Eigen::MatrixXd SVD::getV() const {
    return V;
}
void SVD::setU(const Mat& U__) { U = U__; }

void SVD::setData(const Mat& A) { my_data = A; }
void SVD::compute() {
    
    int dim= my_data.cols() < my_data.rows() ? my_data.cols() : my_data.rows();
    if(full_matrices_){
        U = Mat::Zero(my_data.rows(), my_data.rows());
        V = Mat::Zero(my_data.cols(), my_data.cols());
        S=Vec::Zero(my_data.cols());
    }
    else{
        U = Mat::Zero(my_data.rows(), dim);
        V = Mat::Zero(dim, my_data.cols());
        S=Vec::Zero(dim);
    }
    
    //Mat VT = Mat::Zero(dim,my_data.cols()); // VT is the transpose of V
    

    // Define the matrix B = A^T*A
    //Mat B = my_data.transpose()*my_data; // n*n

    // Define auxiliary vectors u and v
    Vec u = Vec::Zero(my_data.rows());
    Vec v = Vec::Zero(my_data.cols());
    Mat B(my_data.cols(), my_data.cols());
    
    for (unsigned int i=0; i<dim; i++) {
        B = my_data.transpose()*my_data;
        //std::cout << "SVD::compute()" << std::endl;
        PM(my_data, B, S(i), u, v);
        my_data -= S(i)*u*v.transpose();
        U.col(i) = u;
        V.row(i) = v;
    }

    //V = VT.transpose(); // VT is the transpose of V
}

void SVD::rSVD(int &l) {
    // Stage A
    // (1) Form an n × (k + p) Gaussian random matrix Omega
    int m = my_data.rows();
    int n = my_data.cols();

    // int k = 10; // numerical rank (we need an algorithm to find it) or target rank
    // int p = 5; // oversampling parameter, usually it is set to 5 or 10
    // int l = k + p;

    Mat Omega = Mat::Zero(n, l);

    // Create a random number generator for a normal distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    // Fill the corresponding rows of Omega with random numbers using this thread's seed
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < l; ++j) {
            Omega(i, j) = dist(gen);
        }
    }

    int q=1;
    Mat Q = Mat::Zero(m, l);
    intermediate_step(my_data, Q, Omega, l, q);
    // cout << "Q: " << Q << endl;
    // Stage B
    // (4) Form the (k + p) × n matrix B = Q*A
    Mat B = Q.transpose() * my_data;

    // (5) Form the SVD of the small matrix B
    int min= B.rows() < B.cols() ? B.rows() : B.cols();
    Mat Utilde = Mat::Zero(B.rows(), min);
    setData(B);
    setU(Utilde);
    compute();
    // Get Utilde from U
    Utilde = getU();

    // (6) Form U = Q*U_hat
    U = Q * Utilde;
}

void SVD::PM(Mat &A, Mat &B, double &sigma, Vec &u, Vec &v) {
    // Generate a random initial guess x0
    Vec x0 = Vec::Zero(A.cols()); // To multiply B with x0

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (unsigned int i=0; i<x0.size(); i++) {
        x0(i) = distribution(gen);
    }
    x0.normalize();
    // cout << "Check norm of x0: " << x0.norm() << endl;

    // Define the number of iterations
    double epsilon = 1.e-10;
    double delta = 0.05;
    double lambda = 0.1;
    int s = ceil( log(4*log(2*A.cols()/delta)/(epsilon*delta)) /(2*lambda));
    // cout << "Check the number of iterations: " << s << endl;

    for (unsigned int i=1; i<=s; i++) {
        
        x0 = B*x0; // B = A^T*A
        x0.normalize();
    }

    // Compute the left singlular vector
    v = x0;
    v.normalize();

    // Compute the singular value
    Vec temp=A*v;
    sigma = temp.norm();

    // Compute the right singular vector
    u = temp/sigma;
}

void SVD::intermediate_step(Mat &A, Mat &Q, Mat &Omega, int &l, int &q) {
    Mat Y0 = A * Omega; // Y0 = A * Omega = (m*n) * (n*l) = (m*l)
    Mat Q0(A.rows(), l); // Q0 = (m*l)
    Mat R0(l, l); // R0 = (l*l)
    if(full_matrices_){
        qr_decomposition_full(Y0, Q0, R0); 
    }
    else
    qr_decomposition_reduced(Y0, Q0, R0); 

    Mat Ytilde(A.cols(), l); // Ytilde = (n*l)
    Mat Qtilde(A.cols(), l); // Qtilde = (n*l)
    // It is useless to initialize Rtilde because it is still (l*l) and it can be overwritten
    
    for (int j = 1; j <= q; j++) {
        Ytilde = A.transpose() * Q0; // Y0 = A.transpose() * Q0 = (n*m) * (m*l) = (n*l)
        
        if(full_matrices_)
        qr_decomposition_full(Ytilde, Qtilde, R0);
        else
        qr_decomposition_reduced(Ytilde, Qtilde, R0);

        Y0 = A * Qtilde; // Y0 = A * Qtilde = (m*n) * (n*l) = (m*l)
        
        if(full_matrices_)
        qr_decomposition_full(Y0, Q0, R0);
        else
        qr_decomposition_reduced(Y0, Q0, R0);
        
    }
    Q = Q0;
}

void SVD::givens_rotation(double a, double b, Eigen::Matrix2d& G) {
     double r = hypot(a, b);
    double c = a / r;
    double s = -b / r;

    G << c, -s,
         s, c;
}

void SVD::qr_decomposition_full(const Mat &A, Mat &Q, Mat &R) 
{
    int m = A.rows();
    int n = A.cols();
    Eigen::Matrix2d G;

    Q = Mat::Identity(m, m);
    R = A;

    for (int j = 0; j < std::min(m, n); ++j) {
        for (int i = m - 1; i > j; --i) {
            if (R(i, j) != 0) {
                givens_rotation(R(i - 1, j), R(i, j), G);
                R.block(i - 1, j, 2, n - j) = G * R.block(i - 1, j, 2, n - j);
                Q.leftCols(m).middleCols(i - 1, 2) *= G.transpose();

            }
        }
    }
}
void SVD::qr_decomposition_reduced(const Mat &A, Mat &Q, Mat &R) 
{
    int m = A.rows();
    int n = A.cols();
    Eigen::Matrix2d G;

    Mat Q_temp = Mat::Identity(m, m);
    Mat R_temp = A;

    for (int j = 0; j < std::min(m, n); ++j) {
        for (int i = m - 1; i > j; --i) {
            if (R_temp(i, j) != 0) {
                givens_rotation(R_temp(i - 1, j), R_temp(i, j), G);
                
                // R_temp.block(i - 1, j, 2, n - j) = manualMatrixMultiply(G, R_temp.block(i - 1, j, 2, n - j));
                // this matrix block has 2 rows and n-j cols
                int block_rows = 2;
                int block_cols = n-j;
                Mat temp_1(block_rows, block_cols);
                for (int inner_i=0; inner_i<block_rows; ++inner_i){
                    for (int inner_j=0; inner_j<block_cols; ++inner_j){
                        double sum = 0.0;
                        for (int k = 0; k<block_rows; ++k){
                            sum += G(inner_i, k) * R_temp.block(i - 1, j, 2, n - j)(k, inner_j);
                        }
                        temp_1(inner_i, inner_j) = sum;
                    }
                }

                R_temp.block(i - 1, j, 2, n - j) = temp_1;
                Q_temp.leftCols(m).middleCols(i - 1, 2) = Q_temp.leftCols(m).middleCols(i - 1, 2)* G.transpose();

            }
        }
    }
    Q = Q_temp.leftCols(n);
    R = R_temp.topRows(n);
}