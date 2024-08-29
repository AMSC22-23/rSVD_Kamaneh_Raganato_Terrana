#include "SVD_class.hpp"


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
        S=Vec::Zero(my_data.rows());
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
    B = my_data.transpose()*my_data;
    for (unsigned int i=0; i<dim; i++) {
        //B = my_data.transpose()*my_data;
        //std::cout << "SVD::compute()" << std::endl;
        PM(my_data, B, S(i), u, v);
        Mat update = S(i) * u * v.transpose();
    my_data -= update;

    // Aggiorna B
    B -= update.transpose() * update;
        //my_data -= S(i)*u*v.transpose();
        U.col(i) = u;
        V.row(i) = v;
    }

    //V = VT.transpose(); // VT is the transpose of V
}

/*void SVD::rSVD(int &l) {
    // Stage A
    // (1) Form an n × (k + p) Gaussian random matrix Omega
    int m = my_data.rows();
    int n = my_data.cols();

    // int k = 10; // numerical rank (we need an algorithm to find it) or target rank
    // int p = 5; // oversampling parameter, usually it is set to 5 or 10
    // int l = k + p;

    Mat Omega = Mat::Zero(n, l);

    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < l; ++j) {
            Omega(i, j) = dist(gen);
        }
    }

    int q=1;
    Mat Q = Mat::Zero(m, l);
    intermediate_step(my_data, Q, Omega, l, q);
    
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
}*/

void SVD::rSVD(int &l,int &rank,int &num_procs) {
    // Stage A
    // (1) Form an n × (k + p) Gaussian random matrix Omega
    int m = my_data.rows();
    int n = my_data.cols();

    // int k = 10; // numerical rank (we need an algorithm to find it) or target rank
    // int p = 5; // oversampling parameter, usually it is set to 5 or 10
    // int l = k + p;

    Eigen::MatrixXd Omega=Eigen::MatrixXd::Constant(m, n,0.0); 
    
    int rows_per_proc = Omega.rows() / num_procs;
    int rows_remainder = Omega.rows() % num_procs;
    int local_rows = (rank < rows_remainder) ? rows_per_proc + 1 : rows_per_proc;
    int offset = rank * rows_per_proc + std::min(rank, rows_remainder);

    // Initialization of the random number generator on each process
    std::random_device rd;
    //use the rank of the process as the seed
    std::mt19937 gen(rd() + rank); 
    std::normal_distribution<double> dist(0.0, 1.0);

    // Generate local portion of the matrix Omega
    Eigen::MatrixXd local_Omega = Omega.block(offset, 0, local_rows, Omega.cols());
    for (int i = 0; i < local_Omega.rows(); ++i) {
        for (int j = 0; j < n; ++j) {
            local_Omega(i, j) = dist(gen);
        }
    }
    
    std::vector<int> recvcounts(num_procs);
    std::vector<int> displacements(num_procs);

    // Compute displacements and recvcounts arrays
    for (int i = 0; i < num_procs; ++i) {
        recvcounts[i] = (i < rows_remainder) ? (rows_per_proc + 1)*n : rows_per_proc*n;
        displacements[i] = (i < rows_remainder) ? (n * (i * (rows_per_proc + 1))) : (n * (rows_remainder * (rows_per_proc + 1)) + (i - rows_remainder) * n * rows_per_proc);
        
}
    // Gather local results using MPI_Gatherv
    MPI_Gatherv(local_Omega.data(), n*local_rows, MPI_DOUBLE, Omega.data(), recvcounts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank==0){
    int q=1;
    Mat Q = Mat::Zero(m, l);
    intermediate_step(my_data, Q, Omega, l, q);
    
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

void SVD::givens_rotation(double &a, double &b, Eigen::Matrix2d& G) {
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
void SVD::Jacobi_Rotation(Mat& A, double &s, double &tau, int &i, int &j, int &k, int &l) {
    double g = A(i, j);
    double h = A(k, l);
    A(i, j) = g - s * (h + g * tau);
    A(k, l) = h + s * (g - h * tau);
}

void SVD::JacobiSVD(Mat &A) {
    
    int n = A.rows();

    
    U = Mat::Identity(n, n);
    
    Vec d= Vec::Zero(n);
    
    int nrot = 0;
    

    for (int i = 0; i < 50; ++i) {
        double sm = 0.0;
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                sm += std::abs(A(p, q));
            }
        }
        
        if (sm == 0.0) {
            break;
        }

        double tresh = (i < 4) ? 0.2 * sm / (n * n) : 0.0;

        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double g = 100.0 * std::abs(A(p, q));
                if (i > 4 && g <= 1e-8 * std::abs(A(p, p)) && g <= 1e-8 * std::abs(A(q, q))) {
                    A(p, q) = 0.0;
                } else if (std::abs(A(p, q)) > tresh) {
                    double h = A(q, q) - A(p, p);
                    double t;
                    if (g <= 1e-8 * std::abs(h)) {
                        t = A(p, q) / h;
                    } else {
                        double theta = 0.5 * h / A(p, q);
                        t = 1.0 / (std::abs(theta) + std::sqrt(1.0 + theta * theta));
                        if (theta < 0.0) {
                            t = -t;
                        }
                    }

                    double c = 1.0 / std::sqrt(1 + t * t);
                    double s = t * c;
                    double tau = s / (1.0 + c);
                    h = t * A(p, q);
                    A(p, p) -= h;
                    A(q, q) += h;
                    A(p, q) = 0.0;

                    for (int j = 0; j < p; ++j) {
                        Jacobi_Rotation(A, s, tau, j, p, j, q);
                    }
                    //std::cout<<"check 3"<<std::endl;
                    for (int j = p + 1; j < q; ++j) {
                        Jacobi_Rotation(A, s, tau, p, j, j, q);
                    }
                    //std::cout<<"check 4"<<std::endl;
                    for (int j = q + 1; j < n; ++j) {
                        Jacobi_Rotation(A, s, tau, p, j, q, j);
                    }
                    //std::cout<<"check 5"<<std::endl;
                    for (int j = 0; j < n; ++j) {
                        Jacobi_Rotation(U, s, tau, j, p, j, q);
                        
                    }
                    //std::cout<<"check 6"<<std::endl;
                    ++nrot;
                }
            }
        }
    }
    
     d = A.diagonal();
     
     for(int i=0;i<n;i++){
         if(d[i]<0){
             d[i]=-d[i];
             
         }
         
     }
    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(n, 0, n-1);
    std::sort(indices.data(), indices.data() + indices.size(), [&d](int a, int b) { return d[a] > d[b]; });
    Mat U_sorted = Mat(n, n);
    Vec d_sorted = Vec(n);
    for (int i = 0; i < n; ++i) {
        U_sorted.col(i) = U.col(indices[i]);
        d_sorted[i] = d[indices[i]];
    }
    S = d_sorted;
    U = U_sorted;
    
}