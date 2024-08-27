#include "POD.hpp"

// Default constructor for standard POD
POD::POD(Mat_m &S, const int r, const double tol)
{
    cout << "===================================================================" << endl;
    cout << "Default constructor for POD" << endl << endl;
    W = standard_POD(S, r, tol);
}

// Constructor for energy POD
POD::POD(Mat_m &S, Mat_m &Xh, const int r, const double tol)
{
    cout << "===================================================================" << endl;
    cout << "Constructor for energy_POD" << endl << endl;
    W = energy_POD(S, Xh, r, tol);
}

// Constructor for weight POD
POD::POD(Mat_m &S, Mat_m &Xh, Mat_m &D, const int r, const double tol)
{
    cout << "===================================================================" << endl;
    cout << "Constructor for weight_POD" << endl << endl;
    W = weight_POD(S, Xh, D, r, tol);
}

// Constructor for online POD through incremental SVD: starting from A it computes U, Sigma, V
POD::POD(Mat_m &A, Mat_m &U, Mat_m &Sigma, Mat_m &V, const int dim, const Vec_v c, const int M, const int r, const double tol, const double tol_sv)
{   
    cout << "===================================================================" << endl;
    cout << "Constructor for online POD" << endl << endl;
    Vec_v sigma = Vec_v::Zero(dim);
    SVD<SVDMethod::Jacobi> svd(A); // (A, sigma, U, V, dim);
    svd.compute();
    U = svd.getU();
    sigma = svd.getS();
    Sigma = sigma.asDiagonal();
    V = svd.getV();
}

// Power Method
// void POD::PM(Mat_m &A, Mat_m &B, double &sigma, Vec_v &u, Vec_v &v)
// {
//     // Generate a random initial guess x0
//     Vec_v x0 = Vec_v::Zero(A.cols());

//     random_device rd;
//     mt19937 gen(rd());
//     normal_distribution<double> distribution(0.0, 1.0);

//     for (unsigned int i=0; i<x0.size(); i++) {
//         x0(i) = distribution(gen);
//     }
//     x0.normalize();

//     // Define the number of iterations
//     double epsilon = 1.e-10;
//     double delta = 0.05;
//     double lambda = 0.1;
//     int s = ceil( log(4*log(2*A.cols()/delta)/(epsilon*delta)) /(2*lambda));
//     // cout << "Check the number of iterations: " << s << endl;

//     for (unsigned int i=1; i<=s; i++) {
//         x0 = B*x0; // B = A^T*A
//         x0.normalize();
//     }

//     // Compute the left singlular vector
//     v = x0;
//     v.normalize();

//     // Compute the singular value
//     sigma = (A*v).norm();

//     // Compute the right singular vector
//     u = A*v/sigma;
// }

// Singular Value Decomposition through Power Method
// void POD::SVD(Mat_m &A, Vec_v &sigma, Mat_m &U, Mat_m &V, const int dim)
// {
//     Mat_m VT = Mat_m::Zero(dim, A.cols()); // VT is the transpose of V

//     // Define the matrix B = A^T*A
//     Mat_m B = A.transpose()*A; // n*n

//     // Define auxiliary vectors u and v
//     Vec_v u = Vec_v::Zero(A.rows());
//     Vec_v v = Vec_v::Zero(A.cols());
    
//     for (unsigned int i=0; i<dim; i++) {
//         PM(A, B, sigma(i), u, v);
//         A -= sigma(i)*u*v.transpose();
//         B = A.transpose()*A;
//         U.col(i) = u;
//         VT.row(i) = v;
//     }

//     V = VT.transpose(); // V is the transpose of VT
// }

// Algorithm 6.1 page 126 – POD Algorithm
Mat_m POD::standard_POD(Mat_m &S, const int r, const double tol)
{
    cout << "===================================================================" << endl;
    cout << "Standard POD" << endl << endl;
    // NOTE: in the book, W is called V, V is called Z
    // NOTE: it can be proved that solving an eigenvalue problem is equivalent to computing the SVD

    int Nh = S.rows();
    int ns = S.cols();

    // Initialize the matrix W to store all the POD modes, then it will be resized in order to form the POD basis
    // Mat_m W = Mat_m::Zero(Nh, r);
    W = Mat_m::Zero(Nh, r);

    // Initialize the vector sigma to store the singular values
    Vec_v sigma = Vec_v::Zero(r);

    if (ns <= Nh) {
        // Form the correlation matrix
        Mat_m C = S.transpose()*S; // ns*ns
        cout << "Case ns = " << ns << " <= " << Nh << " = Nh" << endl;
        cout << "Check dimensions of C:     " << C.rows() << " * " << C.cols() << endl;

        // Solve the eigenvalue problem through SVD
        // – Initialize the inputs required by the SVD method
        // – Note that the SVD method returns sigma as a vector, then it has to be converted into a diagonal matrix
        Mat_m U = Mat_m::Zero(ns, r);
        Mat_m V = Mat_m::Zero(ns, r);

        // SVD(C, sigma, U, V, r); // i = 1, ..., r
        SVD<SVDMethod::Jacobi> svd(C);
        svd.compute();
        U = svd.getU();
        sigma = svd.getS();
        // Sigma = sigma.asDiagonal();
        V = svd.getV();
        cout << "Check dimensions of U:     " << U.rows() << " * " << U.cols() << endl;
        cout << "Check dimensions of sigma: " << sigma.size() << endl;
        cout << "Check dimensions of V:     " << V.rows() << " * " << V.cols() << endl;

        // Store the POD modes in W 
        for (int i=0; i<r; i++) {
            W.col(i) = S*V.col(i)/sigma(i); // (Nh*ns)*(ns*1) = Nh*1
        }
    }
    else {
        // Form the matrix K
        Mat_m K = S*S.transpose(); // Nh*Nh
        cout << "Case ns = " << ns << " > " << Nh << " = Nh" << endl;
        cout << "Check dimensions of K:     " << K.rows() << " * " << K.cols() << endl;

        // Solve the eigenvalue problem through SVD
        // – Initialize the inputs required by the SVD method
        // – Note that the SVD method returns sigma as a vector, then it has to be converted into a diagonal matrix
        Mat_m U = Mat_m::Zero(Nh, r);
        Mat_m V = Mat_m::Zero(Nh, r);

        // SVD(K, sigma, U, V, r); // i = 1, ..., r
        SVD<SVDMethod::Jacobi> svd(K);
        svd.compute();
        U = svd.getU();
        sigma = svd.getS();
        // Sigma = sigma.asDiagonal();
        V = svd.getV();
        cout << "Check dimensions of U:     " << U.rows() << " * " << U.cols() << endl;
        cout << "Check dimensions of sigma: " << sigma.size() << endl;
        cout << "Check dimensions of V:     " << V.rows() << " * " << V.cols() << endl;

        // Store the POD modes in W
        W = U; // Nh*r
    }

    cout << "Check dimensions of W:     " << W.rows() << " * " << W.cols() << endl;
    // cout << "Check intermediate W in which all POD modes are stored: " << endl << W << endl << endl;

    // Use the criterion to define the minimal POD dimension N <= r such that the projection error is smaller than the
    // desired tolerance
    int N = 0;
    double I = 0.0;
    double num = 0.0;
    Vec_v sigma2 = Vec_v::Zero(r);
    for (int i=0; i<r; i++) {
        sigma2(i) = std::pow(sigma(i), 2);
    }
    double den = sigma2.sum();

    while(I < (1-std::pow(tol, 2)) && (N < r)) {
        num += sigma2(N);
        I = num/den;
        N++;
    }

    // CAPIRE SE N GIUSTO O INCREMENTI SBAGLIATI








    // Build the POD basis
    W.conservativeResize(Eigen::NoChange, N);
    cout << "Check dimensions of W:     " << W.rows() << " * " << W.cols() << endl;
    // cout << "Check final W in which the POD basis is stored: " << endl << W << endl << endl;

    return W;
}

// Algorithm 6.2 page 128 – POD Algorithm with energy norm
Mat_m POD::energy_POD(Mat_m &S, Mat_m &Xh, const int r, const double tol)
{
    cout << "===================================================================" << endl;
    cout << "POD with energy norm" << endl << endl;
    // NOTE: in the book, W is called V, V is called Z
    // NOTE: it can be proved that solving an eigenvalue problem is equivalent to computing the SVD

    int Nh = S.rows();
    int ns = S.cols();

    // Initialize the matrix W to store all the POD modes, then it will be resized in order to form the POD basis
    // Mat_m W = Mat_m::Zero(Nh, r);
    W = Mat_m::Zero(Nh, r);

    // Initialize the vector sigma to store the singular values
    Vec_v sigma = Vec_v::Zero(r);

    if (ns <= Nh) {
        // Form the correlation matrix
        Mat_m Ctilde = (S.transpose()*Xh) * S; // Ctilde = (ns*Nh) * (Nh*Nh) * (Nh*ns) = ns*ns
        cout << "Case ns = " << ns << " <= " << Nh << " = Nh" << endl;
        cout << "Check dimensions of Ctilde: " << Ctilde.rows() << " * " << Ctilde.cols() << endl;

        // Solve the eigenvalue problem through SVD
        // – Initialize the inputs required by the SVD method
        // – Note that the SVD method returns sigma as a vector, then it has to be converted into a diagonal matrix
        Mat_m Utilde = Mat_m::Zero(ns, r);
        Mat_m Vtilde = Mat_m::Zero(ns, r);

        // SVD(Ctilde, sigma, Utilde, Vtilde, r); // i = 1, ..., r
        SVD<SVDMethod::Jacobi> svd(Ctilde);
        svd.compute();
        Utilde = svd.getU();
        sigma = svd.getS();
        // Sigma = sigma.asDiagonal();
        Vtilde = svd.getV();
        cout << "Check dimensions of Utilde: " << Utilde.rows() << " * " << Utilde.cols() << endl;
        cout << "Check dimensions of sigma:  " << sigma.size() << endl;
        cout << "Check dimensions of Vtilde: " << Vtilde.rows() << " * " << Vtilde.cols() << endl;

        // Store the POD modes in W 
        for (int i=0; i<r; i++) {
            W.col(i) = S*Vtilde.col(i)/sigma(i); // (Nh*ns)*(ns*1) = Nh*1
        }
    }
    else {
        // Form the matrix Ktilde
        // NOTE: Xh is a symmetric positive definite matrix, so it is better to use SelfAdjointEigenSolver::operatorSqrt()
        // as explained in https://eigen.tuxfamily.org/dox/unsupported/group__MatrixFunctions__Module.html#matrixbase_sqrt
        // and https://eigen.tuxfamily.org/dox-devel/classEigen_1_1SelfAdjointEigenSolver.html#title20
        Eigen::SelfAdjointEigenSolver<Mat_m> es(Xh);
        Mat_m Xh_sqrt = es.operatorSqrt();
        // cout << "Check Xh: " << endl << Xh << endl << endl;
        // cout << "The square root of Xh is: " << endl << Xh_sqrt << endl << endl;
        // cout << "Check if Xh = Xh_sqrt * Xh_sqrt: " << endl << Xh_sqrt*Xh_sqrt << endl << endl;
        cout << "Norm of ||Xh - Xh_sqrt * Xh_sqrt||: " << (Xh-Xh_sqrt*Xh_sqrt).norm() << endl << endl;

        Mat_m Ktilde = ((Xh_sqrt*S) * S.transpose()) * Xh_sqrt; // Ktilde = (Nh*Nh) * (Nh*ns) * (ns*Nh) * (Nh*Nh) = Nh*Nh
        cout << "Case ns = " << ns << " > " << Nh << " = Nh" << endl;
        cout << "Check dimensions of Ktilde: " << Ktilde.rows() << " * " << Ktilde.cols() << endl;

        // Solve the eigenvalue problem through SVD
        // – Initialize the inputs required by the SVD method
        // – Note that the SVD method returns sigma as a vector, then it has to be converted into a diagonal matrix
        Mat_m Utilde = Mat_m::Zero(Nh, r);
        Mat_m Vtilde = Mat_m::Zero(Nh, r);

        // SVD(Ktilde, sigma, Utilde, Vtilde, r); // i = 1, ..., r
        SVD<SVDMethod::Jacobi> svd(Ktilde);
        svd.compute();
        Utilde = svd.getU();
        sigma = svd.getS();
        // Sigma = sigma.asDiagonal();
        Vtilde = svd.getV();
        cout << "Check dimensions of Utilde: " << Utilde.rows() << " * " << Utilde.cols() << endl;
        cout << "Check dimensions of sigma:  " << sigma.size() << endl;
        cout << "Check dimensions of Vtilde: " << Vtilde.rows() << " * " << Vtilde.cols() << endl;

        // Solve the linear sistem Xh_square * U.col(i) = Utilde.col(i) with Eigen native CG
        Mat_m U = Mat_m::Zero(Nh, r);

        Eigen::ConjugateGradient<Mat_m, Eigen::Lower|Eigen::Upper> cg;
        cg.setMaxIterations(1000);
        cg.setTolerance(1e-12);
        cg.compute(Xh_sqrt);
        for (int i=0; i<Utilde.cols(); i++) {
            U.col(i) = cg.solve(Utilde.col(i));
        }

        // Store the POD modes in W
        W = U; // Nh*r
    }

    cout << "Check dimensions of W:      " << W.rows() << " * " << W.cols() << endl;
    // cout << "Check intermediate W in which all POD modes are stored: " << endl << W << endl << endl;

    // Use the criterion to define the minimal POD dimension N <= r such that the projection error is smaller than the
    // desired tolerance
    int N = 0;
    double I = 0.0;
    double num = 0.0;
    Vec_v sigma2 = Vec_v::Zero(r);
    for (int i=0; i<r; i++) {
        sigma2(i) = std::pow(sigma(i), 2);
    }
    double den = sigma2.sum();

    while(I < (1-std::pow(tol, 2)) && (N < r)) {
        num += sigma2(N);
        I = num/den;
        N++;
    }

    // CAPIRE SE N GIUSTO O INCREMENTI SBAGLIATI








    // Build the POD basis
    W.conservativeResize(Eigen::NoChange, N);
    cout << "Check dimensions of W:      " << W.rows() << " * " << W.cols() << endl;
    // cout << "Check final W in which the POD basis is stored: " << endl << W << endl << endl;

    return W;
}

// Algorithm 6.3 page 134 – POD Algorithm with energy norm and quadrature weights
Mat_m POD::weight_POD(Mat_m &S, Mat_m &Xh, Mat_m &D, const int r, const double tol)
{
    cout << "===================================================================" << endl;
    cout << "POD with energy norm and quadrature weights" << endl << endl;
    // NOTE: in the book, W is called V, V is called Z
    // NOTE: it can be proved that solving an eigenvalue problem is equivalent to computing the SVD

    int Nh = S.rows();
    int ns = S.cols();

    // Initialize the matrix W to store all the POD modes, then it will be resized in order to form the POD basis
    // Mat_m W = Mat_m::Zero(Nh, r);
    W = Mat_m::Zero(Nh, r);

    // Initialize the vector sigma to store the singular values
    Vec_v sigma = Vec_v::Zero(r);

    if (ns <= Nh) {
        // Form the matrix Stilde
        // NOTE: assume that D is a symmetric positive definite matrix, so it is better to use SelfAdjointEigenSolver::operatorSqrt()
        // as explained in https://eigen.tuxfamily.org/dox/unsupported/group__MatrixFunctions__Module.html#matrixbase_sqrt
        // and https://eigen.tuxfamily.org/dox-devel/classEigen_1_1SelfAdjointEigenSolver.html#title20
        Eigen::SelfAdjointEigenSolver<Mat_m> esD(D);
        Mat_m D_sqrt = esD.operatorSqrt();
        // cout << "Check D: " << endl << D << endl << endl;
        // cout << "The square root of D is: " << endl << D_sqrt << endl << endl;
        // cout << "Check if D = D_sqrt * D_sqrt: " << endl << D_sqrt*D_sqrt << endl << endl;
        cout << "Norm of ||D - D_sqrt * D_sqrt||: " << (D-D_sqrt*D_sqrt).norm() << endl << endl;

        Mat_m Stilde = S * D_sqrt; // Stilde = (Nh*ns) * (ns*ns) = Nh*ns

        // Form the correlation matrix
        Mat_m Ctilde = (Stilde.transpose()*Xh) * Stilde; // Ctilde = (ns*Nh) * (Nh*Nh) * (Nh*ns) = ns*ns
        cout << "Case ns = " << ns << " <= " << Nh << " = Nh" << endl;
        cout << "Check dimensions of Ctilde: " << Ctilde.rows() << " * " << Ctilde.cols() << endl;

        // Solve the eigenvalue problem through SVD
        // – Initialize the inputs required by the SVD method
        // – Note that the SVD method returns sigma as a vector, then it has to be converted into a diagonal matrix
        Mat_m Utilde = Mat_m::Zero(ns, r);
        Mat_m Vtilde = Mat_m::Zero(ns, r);

        // SVD(Ctilde, sigma, Utilde, Vtilde, r); // i = 1, ..., r
        SVD<SVDMethod::Jacobi> svd(Ctilde);
        svd.compute();
        Utilde = svd.getU();
        sigma = svd.getS();
        // Sigma = sigma.asDiagonal();
        Vtilde = svd.getV();
        cout << "Check dimensions of Utilde: " << Utilde.rows() << " * " << Utilde.cols() << endl;
        cout << "Check dimensions of sigma:  " << sigma.size() << endl;
        cout << "Check dimensions of Vtilde: " << Vtilde.rows() << " * " << Vtilde.cols() << endl;

        // Store the POD modes in W 
        for (int i=0; i<r; i++) {
            W.col(i) = Stilde*Vtilde.col(i)/sigma(i); // (Nh*ns)*(ns*1) = Nh*1
        }
    }
    else {
        // Form the matrix Ktilde
        // NOTE: Xh is a symmetric positive definite matrix, so it is better to use SelfAdjointEigenSolver::operatorSqrt()
        // as explained in https://eigen.tuxfamily.org/dox/unsupported/group__MatrixFunctions__Module.html#matrixbase_sqrt
        // and https://eigen.tuxfamily.org/dox-devel/classEigen_1_1SelfAdjointEigenSolver.html#title20
        Eigen::SelfAdjointEigenSolver<Mat_m> es(Xh);
        Mat_m Xh_sqrt = es.operatorSqrt();
        // cout << "Check Xh: " << endl << Xh << endl << endl;
        // cout << "The square root of Xh is: " << endl << Xh_sqrt << endl << endl;
        // cout << "Check if Xh = Xh_sqrt * Xh_sqrt: " << endl << Xh_sqrt*Xh_sqrt << endl << endl;
        cout << "Norm of ||Xh - Xh_sqrt * Xh_sqrt||: " << (Xh-Xh_sqrt*Xh_sqrt).norm() << endl << endl;

        Mat_m Ktilde = (((Xh_sqrt*S) * D) * S.transpose()) * Xh_sqrt;
        // Ktilde = (Nh*Nh) * (Nh*ns) * (ns*ns) * (ns*Nh) * (Nh*Nh) = Nh*Nh
        cout << "Case ns = " << ns << " > " << Nh << " = Nh" << endl;
        cout << "Check dimensions of Ktilde: " << Ktilde.rows() << " * " << Ktilde.cols() << endl;

        // Solve the eigenvalue problem through SVD
        // – Initialize the inputs required by the SVD method
        // – Note that the SVD method returns sigma as a vector, then it has to be converted into a diagonal matrix
        Mat_m Utilde = Mat_m::Zero(Nh, r);
        Mat_m Vtilde = Mat_m::Zero(Nh, r);

        // SVD(Ktilde, sigma, Utilde, Vtilde, r); // i = 1, ..., r
        SVD<SVDMethod::Jacobi> svd(Ktilde);
        svd.compute();
        Utilde = svd.getU();
        sigma = svd.getS();
        // Sigma = sigma.asDiagonal();
        Vtilde = svd.getV();
        cout << "Check dimensions of Utilde: " << Utilde.rows() << " * " << Utilde.cols() << endl;
        cout << "Check dimensions of sigma:  " << sigma.size() << endl;
        cout << "Check dimensions of Vtilde: " << Vtilde.rows() << " * " << Vtilde.cols() << endl;

        // Solve the linear sistem Xh_square * U.col(i) = Utilde.col(i) with Eigen native CG
        Mat_m U = Mat_m::Zero(Nh, r);

        Eigen::ConjugateGradient<Mat_m, Eigen::Lower|Eigen::Upper> cg;
        cg.setMaxIterations(1000);
        cg.setTolerance(1e-12);
        cg.compute(Xh_sqrt);
        for (int i=0; i<Utilde.cols(); i++) {
            U.col(i) = cg.solve(Utilde.col(i));
        }

        // Store the POD modes in W
        W = U; // Nh*r
    }

    cout << "Check dimensions of W:      " << W.rows() << " * " << W.cols() << endl;
    // cout << "Check intermediate W in which all POD modes are stored: " << endl << W << endl << endl;

    // Use the criterion to define the minimal POD dimension N <= r such that the projection error is smaller than the
    // desired tolerance
    int N = 0;
    double I = 0.0;
    double num = 0.0;
    Vec_v sigma2 = Vec_v::Zero(r);
    for (int i=0; i<r; i++) {
        sigma2(i) = std::pow(sigma(i), 2);
    }
    double den = sigma2.sum();

    while(I < (1-std::pow(tol, 2)) && (N < r)) {
        num += sigma2(N);
        I = num/den;
        N++;
    }

    // CAPIRE SE N GIUSTO O INCREMENTI SBAGLIATI








    // Build the POD basis
    W.conservativeResize(Eigen::NoChange, N);
    cout << "Check dimensions of W:      " << W.rows() << " * " << W.cols() << endl;
    // cout << "Check final W in which the POD basis is stored: " << endl << W << endl << endl;

    return W;
}

// Standard incremental SVD for building POD – Algorithm 1
void POD::standard_iSVD(Mat_m &U, Mat_m &Sigma, Mat_m &V, const Vec_v c, const double tol, const double tol_sv)
{
    cout << "===================================================================" << endl;
    cout << "Standard incremental SVD" << endl << endl;
    // NOTE: in the paper, U is called V and V is called W
    // NOTE: in the paper, the starting index is 1, here it is 0

    // Initial dimensions:
    // U       n*k
    // Sigma   k*k
    // V       k*k
    // c       n*1

    int n = U.rows();
    int k = U.cols();

    // Step 1: projection
    Vec_v d = U.transpose() * c; // d = U^T*c = (k*n)(n*1) = k*1
    double p = std::sqrt( ((c-U*d).transpose() * (c-U*d)).norm() ); // p is a scalar

    // Initialize the matrix Q
    Mat_m Q = Mat_m::Zero(k+1, k+1);
    Q.block(0, 0, k, k) = Sigma;
    Q.block(0, k, k, 1) = d;
    if (p < tol)
        Q(k, k) = p;
    cout << "Check Sigma:" << endl << Sigma << endl << endl;
    cout << "Check d:" << endl << d << endl << endl;
    cout << "Check p:" << endl << p << endl << endl;
    cout << "Check Q:" << endl << Q << endl << endl;

    // Step 2: SVD solution
    // – Initialize the inputs required by the SVD method
    // – Note that the SVD method returns sigmaQ as a vector, then it has to be converted into a diagonal matrix
    Vec_v sigmaQ = Vec_v::Zero(k+1);
    Mat_m SigmaQ = Mat_m::Zero(k+1, k+1);
    Mat_m UQ = Mat_m::Zero(k+1, k+1);
    Mat_m VQ = Mat_m::Zero(k+1, k+1);

    // SVD(Q, sigmaQ, UQ, VQ, k+1);
    SVD<SVDMethod::Jacobi> svd(Q); // (C, sigma, U, V, dim);
    svd.compute();
    UQ = svd.getU();
    sigmaQ = svd.getS();
    // Sigma = sigma.asDiagonal();
    VQ = svd.getV();
    SigmaQ = sigmaQ.asDiagonal();
    cout << "Check UQ:" << endl << UQ << endl << endl;
    cout << "Check SigmaQ:" << endl << SigmaQ << endl << endl;
    cout << "Check VQ:" << endl << VQ << endl << endl;

    cout << "Compare intermediate and final dimensions with the initial k = " << k << endl << endl;

    // Step 3: left singular vectors update
    // Decision: will the added column increase the rank of the updated matrix?
    if (p < tol || k >= n) {
        U = U * UQ.block(0, 0, k, k);

        Sigma = SigmaQ.block(0, 0, k, k);

        // Auxiliary matrix
        Mat_m Vaux = Mat_m::Zero(k+1, k+1);
        Vaux.block(0, 0, k, k) = V;
        Vaux(k, k) = 1;

        V.conservativeResize(k+1, k);
        V = Vaux * VQ.block(0, 0, k+1, k); // V = (k+1)*(k+1) * (k+1)*k = (k+1)*k

        // Intermediate dimensions:
        // U       n*k
        // Sigma   k*k
        // V       (k+1)*k
    }
    else {
        Vec_v j = (c - U*d) / p; // j = (n*1) - (n*k)*(k*1) = n*1
        U.conservativeResize(n, k+1);
        U.col(k) = j;
        U = U * UQ; // U = n*(k+1) * (k+1)*(k+1) = n*(k+1)

        Sigma.conservativeResize(k+1, k+1);
        Sigma = SigmaQ;

        // In this case, the auxiliary matrix Vaux is not needed
        V.conservativeResize(k+1, k+1);
        V(k, k) = 1;
        V = V * VQ; // V = (k+1)*(k+1) * (k+1)*(k+1) = (k+1)*(k+1)

        k++;

        // Intermediate dimensions with respect to the initial k=U.cols():
        // U       n*(k+1)
        // Sigma   (k+1)*(k+1)
        // V       (k+1)*(k+1)
    }

    cout << "Intermediate dimensions: ";
    if ((p < tol || k >= n) == true)
        cout << "in step 3 the 'if' branch is taken" << endl;
    else
        cout << "in step 3 the 'else' branch is taken" << endl;
    cout << "Check dimensions of U:     " << U.rows() << " * " << U.cols() << endl;
    cout << "Check dimensions of Sigma: " << Sigma.rows() << " * " << Sigma.cols() << endl;
    cout << "Check dimensions of V:     " << V.rows() << " * " << V.cols() << endl << endl;

    // Step 4: SEE THE ENHANCED ALGORITHM

    // Step 5: small SV trucation
    if ((Sigma(k-2, k-2) > tol_sv) && (Sigma(k-1, k-1) < tol_sv)) {
        int kk = k-1;
        U.conservativeResize(Eigen::NoChange, kk);
        Sigma.conservativeResize(kk, kk);
        V.conservativeResize(Eigen::NoChange, kk);

        // Final dimensions:
        // U       n*kk
        // Sigma   kk*kk
        // V       _*kk

        // Final dimensions with respect to the initial k=U.cols(), in case "if" branch is taken in step 3:
        // U       n*(k-1)
        // Sigma   (k-1)*(k-1)
        // V       (k+1)*(k-1)

        // Final dimensions with respect to the initial k=U.cols(), in case "else" branch is taken in step 3:
        // U       n*k
        // Sigma   k*k
        // V       (k+1)*k

        k = kk; // kk is an auxiliary variable to visualize easily the final dimensions
    }

    // Final dimensions with respect to the initial k=U.cols(), in case "if" branch is not taken in step 5 and
    // "if" branch is taken in step 3:
    // U       n*k
    // Sigma   k*k
    // V       (k+1)*k

    // Final dimensions with respect to the initial k=U.cols(), in case "if" branch is not taken in step 5 and
    // "else" branch is taken in step 3:
    // U       n*(k+1)
    // Sigma   (k+1)*(k+1)
    // V       (k+1)*(k+1)

    cout << "Final dimensions: " << endl;
    if (((Sigma(k-2, k-2) > tol_sv) && (Sigma(k-1, k-1) < tol_sv)) == true)
        cout << "in step 5 the 'if' branch is taken" << endl;
    else
        cout << "in step 5 the 'if' branch is not taken" << endl;
    cout << "Check dimensions of U:     " << U.rows() << " * " << U.cols() << endl;
    cout << "Check dimensions of Sigma: " << Sigma.rows() << " * " << Sigma.cols() << endl;
    cout << "Check dimensions of V:     " << V.rows() << " * " << V.cols() << endl << endl;


    // FUNZIONA TOGLIENDO DIRETTAMENTE?


    // FARE PER ROBUSTEZZA O EVITARE?

    // Step 6: reorthogonalization
    // Note that epsilon is a double-precision machine epsilon
    // if ((V.col(k-1).transpose()*V.col(0)).norm() > std::min(tol, n*std::numeric_limits<double>::epsilon())) {
    //     V = modifiedGramSchmidt(V);
    // }


}

// Enhanced incremental SVD for building POD – Algorithm 2
void POD::enhanced_iSVD(Mat_m &U, Mat_m &Sigma, Mat_m &V, const Vec_v c, const int M, const double tol, const double tol_sv)
{
    cout << "===================================================================" << endl;
    cout << "Enhanced incremental SVD" << endl << endl;
    // NOTE: in the paper, U is called V and V is called W
    // NOTE: in the paper, the starting index is 1, here it is 0

    // Initial dimensions:
    // U       n*k
    // Sigma   k*k
    // V       k*k
    // c       n*1

    int n = U.rows();
    int k = U.cols();

    // Step 1: projection
    Vec_v d = U.transpose() * c; // d = U^T*c = (k*n)(n*1) = k*1
    double p = std::sqrt( ((c-U*d).transpose() * (c-U*d)).norm() ); // p is a scalar

    // Initialize the matrix Q
    Mat_m Q = Mat_m::Zero(k+1, k+1);
    Q.block(0, 0, k, k) = Sigma;
    Q.block(0, k, k, 1) = d;
    if (p < tol)
        Q(k, k) = p;
    cout << "Check Sigma:" << endl << Sigma << endl << endl;
    cout << "Check d:" << endl << d << endl << endl;
    cout << "Check p:" << endl << p << endl << endl;
    cout << "Check Q:" << endl << Q << endl << endl;

    // Step 2: SVD solution
    // – Initialize the inputs required by the SVD method
    // – Note that the SVD method returns sigmaQ as a vector, then it has to be converted into a diagonal matrix
    Vec_v sigmaQ = Vec_v::Zero(k+1);
    Mat_m SigmaQ = Mat_m::Zero(k+1, k+1);
    Mat_m UQ = Mat_m::Zero(k+1, k+1);
    Mat_m VQ = Mat_m::Zero(k+1, k+1);

    // SVD(Q, sigmaQ, UQ, VQ, k+1);
    SVD<SVDMethod::Jacobi> svd(Q); // (C, sigma, U, V, dim);
    svd.compute();
    UQ = svd.getU();
    sigmaQ = svd.getS();
    // Sigma = sigma.asDiagonal();
    VQ = svd.getV();
    SigmaQ = sigmaQ.asDiagonal();
    cout << "Check UQ:" << endl << UQ << endl << endl;
    cout << "Check SigmaQ:" << endl << SigmaQ << endl << endl;
    cout << "Check VQ:" << endl << VQ << endl << endl;

    cout << "Compare intermediate and final dimensions with the initial k = " << k << endl << endl;

    // Step 3: left singular vectors update
    // Decision: will the added column increase the rank of the updated matrix?
    if (p < tol || k >= n) {
        U = U * UQ.block(0, 0, k, k);

        Sigma = SigmaQ.block(0, 0, k, k);

        // Auxiliary matrix
        Mat_m Vaux = Mat_m::Zero(k+1, k+1);
        Vaux.block(0, 0, k, k) = V;
        Vaux(k, k) = 1;

        V.conservativeResize(k+1, k);
        V = Vaux * VQ.block(0, 0, k+1, k); // V = (k+1)*(k+1) * (k+1)*k = (k+1)*k

        // Intermediate dimensions:
        // U       n*k
        // Sigma   k*k
        // V       (k+1)*k
    }
    else {
        Vec_v j = (c - U*d) / p; // j = (n*1) - (n*k)*(k*1) = n*1
        U.conservativeResize(n, k+1);
        U.col(k) = j;
        U = U * UQ; // U = n*(k+1) * (k+1)*(k+1) = n*(k+1)

        Sigma.conservativeResize(k+1, k+1);
        Sigma = SigmaQ;

        // In this case, the auxiliary matrix Vaux is not needed
        V.conservativeResize(k+1, k+1);
        V(k, k) = 1;
        V = V * VQ; // V = (k+1)*(k+1) * (k+1)*(k+1) = (k+1)*(k+1)

        k++;

        // Intermediate dimensions with respect to the initial k=U.cols():
        // U       n*(k+1)
        // Sigma   (k+1)*(k+1)
        // V       (k+1)*(k+1)
    }

    cout << "Intermediate dimensions: " << endl;
    if ((p < tol || k >= n) == true)
        cout << "in step 3 the 'if' branch is taken" << endl;
    else
        cout << "in step 3 the 'else' branch is taken" << endl;
    cout << "Check dimensions of U:     " << U.rows() << " * " << U.cols() << endl;
    cout << "Check dimensions of Sigma: " << Sigma.rows() << " * " << Sigma.cols() << endl;
    cout << "Check dimensions of V:     " << V.rows() << " * " << V.cols() << endl << endl;

    // Step 4
    if (k > M) {
        U.conservativeResize(Eigen::NoChange, M);
        Sigma.conservativeResize(M, M);
        V.conservativeResize(Eigen::NoChange, M);
        k = M;
    }

    // Step 5: small SV trucation
    if ((Sigma(k-2, k-2) > tol_sv) && (Sigma(k-1, k-1) < tol_sv)) {
        int kk = k-1;
        U.conservativeResize(Eigen::NoChange, kk);
        Sigma.conservativeResize(kk, kk);
        V.conservativeResize(Eigen::NoChange, kk);

        // Final dimensions:
        // U       n*kk
        // Sigma   kk*kk
        // V       _*kk

        // Final dimensions with respect to the initial k=U.cols(), in case "if" branch is taken in step 3 and
        // "if" branch is not taken in step 4:
        // U       n*(k-1)
        // Sigma   (k-1)*(k-1)
        // V       (k+1)*(k-1)

        // Final dimensions with respect to the initial k=U.cols(), in case "else" branch is taken in step 3 and
        // "if" branch is not taken in step 4:
        // U       n*k
        // Sigma   k*k
        // V       (k+1)*k

        // Final dimensions with respect to the initial k=U.cols(), in case "if" branch is taken in step 4 (kk = M-1):
        // U       n*(M-1)
        // Sigma   (M-1)*(M-1)
        // V       (k+1)*(M-1)

        k = kk; // kk is an auxiliary variable to visualize easily the final dimensions
    }

    // Final dimensions with respect to the initial k=U.cols(), in case "if" branch is not taken in step 5 and
    // "if" branch is taken in step 3 and "if" branch is not taken in step 4:
    // U       n*k
    // Sigma   k*k
    // V       (k+1)*k

    // Final dimensions with respect to the initial k=U.cols(), in case "if" branch is not taken in step 5 and
    // "else" branch is taken in step 3 and "if" branch is not taken in step 4:
    // U       n*(k+1)
    // Sigma   (k+1)*(k+1)
    // V       (k+1)*(k+1)

    // Final dimensions with respect to the initial k=U.cols(), in case "if" branch is not taken in step 5 and
    // and "if" branch is taken in step 4:
    // U       n*M
    // Sigma   M*M
    // V       (k+1)*M

    cout << "Final dimensions: " << endl;
    if (k == M)
        cout << "in step 4 the 'if' branch is taken, " << endl;
    else
        cout << "in step 4 the 'if' branch is not taken, " << endl;
    if (((Sigma(k-2, k-2) > tol_sv) && (Sigma(k-1, k-1) < tol_sv)) == true)
        cout << "in step 5 the 'if' branch is taken" << endl;
    else
        cout << "in step 5 the 'if' branch is not taken" << endl;
    cout << "Check dimensions of U:     " << U.rows() << " * " << U.cols() << endl;
    cout << "Check dimensions of Sigma: " << Sigma.rows() << " * " << Sigma.cols() << endl;
    cout << "Check dimensions of V:     " << V.rows() << " * " << V.cols() << endl << endl;







    // FUNZIONA TOGLIENDO DIRETTAMENTE?


    // FARE PER ROBUSTEZZA O EVITARE?

    // Step 6: reorthogonalization
    // Note that epsilon is a double-precision machine epsilon
    // if ((V.col(k-1).transpose()*V.col(0)).norm() > std::min(tol, n*std::numeric_limits<double>::epsilon())) {
    //     V = modifiedGramSchmidt(V);
    // }
}

