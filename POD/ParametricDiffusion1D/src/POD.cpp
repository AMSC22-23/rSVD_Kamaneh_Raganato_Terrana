#include "POD.hpp"

// Default constructor for POD
POD::POD()
{
    // std::cout << "===================================================================" << std::endl;
    // std::cout << "Default constructor for POD" << std::endl << std::endl;
}

// Constructor for naive POD
POD::POD(Mat_m &S, const int svd_type)
{   
    // std::cout << "===================================================================" << std::endl;
    // std::cout << "Constructor for naive POD" << std::endl << std::endl;
    std::tie(W, sigma) = naive_POD(S, svd_type);
}

// Constructor for standard POD
POD::POD(Mat_m &S, const int r, const double tol, const int svd_type)
{
    // std::cout << "===================================================================" << std::endl;
    // std::cout << "Constructor for standard POD" << std::endl << std::endl;
    std::tie(W, sigma) = standard_POD(S, r, tol, svd_type);
}

// Constructor for energy POD
POD::POD(Mat_m &S, Mat_m &Xh, const int r, const double tol, const int svd_type)
{
    // std::cout << "===================================================================" << std::endl;
    // std::cout << "Constructor for energy POD" << std::endl << std::endl;
    std::tie(W, sigma) = energy_POD(S, Xh, r, tol, svd_type);
}

// Constructor for weight POD
POD::POD(Mat_m &S, Mat_m &Xh, Mat_m &D, const int r, const double tol, const int svd_type)
{
    // std::cout << "===================================================================" << std::endl;
    // std::cout << "Constructor for weight POD" << std::endl << std::endl;
    std::tie(W, sigma) = weight_POD(S, Xh, D, r, tol, svd_type);
}

// NON SO, questo ha molti argomenti ma non ti sembra collegato a quelli sotto, magari dovresti sostituire chiamata a SVD con chiamata a uno di quei due
// Constructor for online POD through incremental SVD: starting from A it computes U, Sigma, V
// POD::POD(Mat_m &A, Mat_m &U, Mat_m &Sigma, Mat_m &V, const int dim, const Vec_v c, const int M, const int r, const double tol, const double tol_sv)
// {   
//     std::cout << "===================================================================" << std::endl;
//     std::cout << "Constructor for online POD" << std::endl << std::endl;
//     Vec_v sigma = Vec_v::Zero(dim);
//     SVD<SVDMethod::Jacobi> svd(A); // (A, sigma, U, V, dim);
//     svd.compute();
//     U = svd.getU();
//     sigma = svd.getS();
//     Sigma = sigma.asDiagonal();
//     V = svd.getV();
// }

void POD::perform_SVD(Mat_m &A, Mat_m &U, Vec_v &sigma, Mat_m &V, const int r, const int svd_type)
{
    std::unique_ptr<SVD<SVDMethod::Power>> svd_power;
    std::unique_ptr<SVD<SVDMethod::Jacobi>> svd_jacobi;
    std::unique_ptr<SVD<SVDMethod::DynamicJacobi>> svd_dynamic_jacobi;
    std::unique_ptr<SVD<SVDMethod::ParallelJacobi>> svd_parallel_jacobi;

    // Note that blocks {} are needed inside each case to allow the declaration of svd.
    switch (svd_type)
    {
        case 0:
        {
            std::cout << "  SVD with Power Method" << std::endl;
            svd_power = std::make_unique<SVD<SVDMethod::Power>>(A, r);
            break;
        }
        case 1:
        {
            std::cout << "  SVD with Jacobi Method" << std::endl;
            svd_jacobi = std::make_unique<SVD<SVDMethod::Jacobi>>(A);
            break;
        }
        case 2:
        {
            std::cout << "  SVD with Dynamic Jacobi Method" << std::endl;
            svd_dynamic_jacobi = std::make_unique<SVD<SVDMethod::DynamicJacobi>>(A);
            break;
        }
        case 3:
        {
            std::cout << "  SVD with Parallel Jacobi Method" << std::endl;
            svd_parallel_jacobi = std::make_unique<SVD<SVDMethod::ParallelJacobi>>(A);
            break;
        }
        default:
        {
            std::cerr << "The svd_type should be among 0, 1, 2, 3. Check 'svd_type' in the parameter file." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    if (svd_power)
    {
        svd_power->compute();
        U = svd_power->getU();
        sigma = svd_power->getS();
        V = svd_power->getV();
    }
    else if (svd_jacobi)
    {
        svd_jacobi->compute();
        U = svd_jacobi->getU();
        sigma = svd_jacobi->getS();
        V = svd_jacobi->getV();
    }
    else if (svd_dynamic_jacobi)
    {
        svd_dynamic_jacobi->compute();
        U = svd_dynamic_jacobi->getU();
        sigma = svd_dynamic_jacobi->getS();
        V = svd_dynamic_jacobi->getV();
    }
    else if (svd_parallel_jacobi)
    {
        svd_parallel_jacobi->compute();
        U = svd_parallel_jacobi->getU();
        sigma = svd_parallel_jacobi->getS();
        V = svd_parallel_jacobi->getV();
    }
}

std::tuple<Mat_m, Vec_v> POD::naive_POD(Mat_m &S, const int svd_type)
{
    std::cout << "===================================================================" << std::endl;
    std::cout << "Naive POD" << std::endl;

    // SISTEMA TUTTI COMMENTI; qui in realtà hai già inizializzato, stai solo ridefinendo dimensioni
    // Initialize the matrix W to store all the POD modes, then it will be resized in order to form the POD basis
    W = Mat_m::Zero(S.rows(), S.rows());

    // Initialize the vector sigma to store the singular values
    int min= S.rows() < S.cols() ? S.rows() : S.cols();
    sigma = Vec_v::Zero(min);

    Mat_m V = Mat_m::Zero(S.cols(), S.cols());

    perform_SVD(S, W, sigma, V, 0, svd_type);

    return std::make_tuple(W, sigma);
}

// Algorithm 6.1 page 126 – POD Algorithm
std::tuple<Mat_m, Vec_v> POD::standard_POD(Mat_m &S, const int r, const double tol, const int svd_type)
{
    std::cout << "===================================================================" << std::endl;
    std::cout << "Standard POD" << std::endl;
    // NOTE: in the book, W is called V, V is called Z
    // NOTE: it can be proved that solving an eigenvalue problem is equivalent to computing the SVD

    int Nh = S.rows();
    int ns = S.cols();

    // Initialize the matrix W to store all the POD modes, then it will be resized in order to form the POD basis
    // Mat_m W = Mat_m::Zero(Nh, r);
    W = Mat_m::Zero(Nh, r);

    // Initialize the vector sigma to store the singular values
    sigma = Vec_v::Zero(r);

    if (ns <= Nh) {
        // Form the correlation matrix
        Mat_m C = S.transpose()*S; // ns*ns
        std::cout << "Case ns = " << ns << " <= " << Nh << " = Nh" << std::endl;
        std::cout << "Check dimensions of C:     " << C.rows() << " * " << C.cols() << std::endl;

        // Solve the eigenvalue problem through SVD
        // – Initialize the inputs required by the SVD method
        // – Note that the SVD method returns sigma as a vector, then it has to be converted into a diagonal matrix
        Mat_m U = Mat_m::Zero(ns, r);
        Mat_m V = Mat_m::Zero(ns, r);

        // SVD(C, sigma, U, V, r); // i = 1, ..., r
        // SVD<SVDMethod::Jacobi> svd(C);
        perform_SVD(C, U, sigma, V, r, svd_type);
        
        std::cout << "Check dimensions of U:     " << U.rows() << " * " << U.cols() << std::endl;
        std::cout << "Check dimensions of sigma: " << sigma.size() << std::endl;
        std::cout << "Check dimensions of V:     " << V.rows() << " * " << V.cols() << std::endl;

        // Store the POD modes in W 
        for (int i=0; i<r; i++) {
            W.col(i) = S*V.col(i)/sigma(i); // (Nh*ns)*(ns*1) = Nh*1
        }
    }
    else {
        // Form the matrix K
        Mat_m K = S*S.transpose(); // Nh*Nh
        std::cout << "Case ns = " << ns << " > " << Nh << " = Nh" << std::endl;
        std::cout << "Check dimensions of K:     " << K.rows() << " * " << K.cols() << std::endl;

        // Solve the eigenvalue problem through SVD
        // – Initialize the inputs required by the SVD method
        // – Note that the SVD method returns sigma as a vector, then it has to be converted into a diagonal matrix
        Mat_m U = Mat_m::Zero(Nh, r);
        Mat_m V = Mat_m::Zero(Nh, r);

        // SVD(K, sigma, U, V, r); // i = 1, ..., r
        // SVD<SVDMethod::Jacobi> svd(K);
        perform_SVD(K, U, sigma, V, r, svd_type);
        
        std::cout << "Check dimensions of U:     " << U.rows() << " * " << U.cols() << std::endl;
        std::cout << "Check dimensions of sigma: " << sigma.size() << std::endl;
        std::cout << "Check dimensions of V:     " << V.rows() << " * " << V.cols() << std::endl;

        // Store the POD modes in W
        W = U; // Nh*r
    }

    std::cout << "Check dimensions of W:     " << W.rows() << " * " << W.cols() << std::endl;
    // std::cout << "Check intermediate W in which all POD modes are stored: " << std::endl << W << std::endl << std::endl;

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
    std::cout << "Check dimensions of W:     " << W.rows() << " * " << W.cols() << std::endl;
    // std::cout << "Check final W in which the POD basis is stored: " << std::endl << W << std::endl << std::endl;

    return std::make_tuple(W, sigma);
}

// Algorithm 6.2 page 128 – POD Algorithm with energy norm
std::tuple<Mat_m, Vec_v> POD::energy_POD(Mat_m &S, Mat_m &Xh, const int r, const double tol, const int svd_type)
{
    std::cout << "===================================================================" << std::endl;
    std::cout << "POD with energy norm" << std::endl;
    // NOTE: in the book, W is called V, V is called Z
    // NOTE: it can be proved that solving an eigenvalue problem is equivalent to computing the SVD

    int Nh = S.rows();
    int ns = S.cols();

    // Initialize the matrix W to store all the POD modes, then it will be resized in order to form the POD basis
    // Mat_m W = Mat_m::Zero(Nh, r);
    W = Mat_m::Zero(Nh, r);

    // Initialize the vector sigma to store the singular values
    sigma = Vec_v::Zero(r);

    if (ns <= Nh) {
        // Form the correlation matrix
        Mat_m Ctilde = (S.transpose()*Xh) * S; // Ctilde = (ns*Nh) * (Nh*Nh) * (Nh*ns) = ns*ns
        std::cout << "Case ns = " << ns << " <= " << Nh << " = Nh" << std::endl;
        std::cout << "Check dimensions of Ctilde: " << Ctilde.rows() << " * " << Ctilde.cols() << std::endl;

        // Solve the eigenvalue problem through SVD
        // – Initialize the inputs required by the SVD method
        // – Note that the SVD method returns sigma as a vector, then it has to be converted into a diagonal matrix
        Mat_m Utilde = Mat_m::Zero(ns, r);
        Mat_m Vtilde = Mat_m::Zero(ns, r);

        // SVD(Ctilde, sigma, Utilde, Vtilde, r); // i = 1, ..., r
        // SVD<SVDMethod::Jacobi> svd(Ctilde);
        perform_SVD(Ctilde, Utilde, sigma, Vtilde, r, svd_type);
    
        std::cout << "Check dimensions of Utilde: " << Utilde.rows() << " * " << Utilde.cols() << std::endl;
        std::cout << "Check dimensions of sigma:  " << sigma.size() << std::endl;
        std::cout << "Check dimensions of Vtilde: " << Vtilde.rows() << " * " << Vtilde.cols() << std::endl;

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
        // std::cout << "Check Xh: " << std::endl << Xh << std::endl << std::endl;
        // std::cout << "The square root of Xh is: " << std::endl << Xh_sqrt << std::endl << std::endl;
        // std::cout << "Check if Xh = Xh_sqrt * Xh_sqrt: " << std::endl << Xh_sqrt*Xh_sqrt << std::endl << std::endl;
        std::cout << "Norm of ||Xh - Xh_sqrt * Xh_sqrt||: " << (Xh-Xh_sqrt*Xh_sqrt).norm() << std::endl << std::endl;

        Mat_m Ktilde = ((Xh_sqrt*S) * S.transpose()) * Xh_sqrt; // Ktilde = (Nh*Nh) * (Nh*ns) * (ns*Nh) * (Nh*Nh) = Nh*Nh
        std::cout << "Case ns = " << ns << " > " << Nh << " = Nh" << std::endl;
        std::cout << "Check dimensions of Ktilde: " << Ktilde.rows() << " * " << Ktilde.cols() << std::endl;

        // Solve the eigenvalue problem through SVD
        // – Initialize the inputs required by the SVD method
        // – Note that the SVD method returns sigma as a vector, then it has to be converted into a diagonal matrix
        Mat_m Utilde = Mat_m::Zero(Nh, r);
        Mat_m Vtilde = Mat_m::Zero(Nh, r);

        // SVD(Ktilde, sigma, Utilde, Vtilde, r); // i = 1, ..., r
        // SVD<SVDMethod::Jacobi> svd(Ktilde);
        perform_SVD(Ktilde, Utilde, sigma, Vtilde, r, svd_type);
     
        std::cout << "Check dimensions of Utilde: " << Utilde.rows() << " * " << Utilde.cols() << std::endl;
        std::cout << "Check dimensions of sigma:  " << sigma.size() << std::endl;
        std::cout << "Check dimensions of Vtilde: " << Vtilde.rows() << " * " << Vtilde.cols() << std::endl;

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

    std::cout << "Check dimensions of W:      " << W.rows() << " * " << W.cols() << std::endl;
    // std::cout << "Check intermediate W in which all POD modes are stored: " << std::endl << W << std::endl << std::endl;

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
    std::cout << "Check dimensions of W:      " << W.rows() << " * " << W.cols() << std::endl;
    // std::cout << "Check final W in which the POD basis is stored: " << std::endl << W << std::endl << std::endl;

    return std::make_tuple(W, sigma);
}

// Algorithm 6.3 page 134 – POD Algorithm with energy norm and quadrature weights
std::tuple<Mat_m, Vec_v> POD::weight_POD(Mat_m &S, Mat_m &Xh, Mat_m &D, const int r, const double tol, const int svd_type)
{
    std::cout << "===================================================================" << std::endl;
    std::cout << "POD with energy norm and quadrature weights" << std::endl;
    // NOTE: in the book, W is called V, V is called Z
    // NOTE: it can be proved that solving an eigenvalue problem is equivalent to computing the SVD

    int Nh = S.rows();
    int ns = S.cols();

    // Initialize the matrix W to store all the POD modes, then it will be resized in order to form the POD basis
    // Mat_m W = Mat_m::Zero(Nh, r);
    W = Mat_m::Zero(Nh, r);

    // Initialize the vector sigma to store the singular values
    sigma = Vec_v::Zero(r);

    if (ns <= Nh) {
        // Form the matrix Stilde
        // NOTE: assume that D is a symmetric positive definite matrix, so it is better to use SelfAdjointEigenSolver::operatorSqrt()
        // as explained in https://eigen.tuxfamily.org/dox/unsupported/group__MatrixFunctions__Module.html#matrixbase_sqrt
        // and https://eigen.tuxfamily.org/dox-devel/classEigen_1_1SelfAdjointEigenSolver.html#title20
        Eigen::SelfAdjointEigenSolver<Mat_m> esD(D);
        Mat_m D_sqrt = esD.operatorSqrt();
        // std::cout << "Check D: " << std::endl << D << std::endl << std::endl;
        // std::cout << "The square root of D is: " << std::endl << D_sqrt << std::endl << std::endl;
        // std::cout << "Check if D = D_sqrt * D_sqrt: " << std::endl << D_sqrt*D_sqrt << std::endl << std::endl;
        std::cout << "Norm of ||D - D_sqrt * D_sqrt||: " << (D-D_sqrt*D_sqrt).norm() << std::endl << std::endl;

        Mat_m Stilde = S * D_sqrt; // Stilde = (Nh*ns) * (ns*ns) = Nh*ns

        // Form the correlation matrix
        Mat_m Ctilde = (Stilde.transpose()*Xh) * Stilde; // Ctilde = (ns*Nh) * (Nh*Nh) * (Nh*ns) = ns*ns
        std::cout << "Case ns = " << ns << " <= " << Nh << " = Nh" << std::endl;
        std::cout << "Check dimensions of Ctilde: " << Ctilde.rows() << " * " << Ctilde.cols() << std::endl;

        // Solve the eigenvalue problem through SVD
        // – Initialize the inputs required by the SVD method
        // – Note that the SVD method returns sigma as a vector, then it has to be converted into a diagonal matrix
        Mat_m Utilde = Mat_m::Zero(ns, r);
        Mat_m Vtilde = Mat_m::Zero(ns, r);

        // SVD(Ctilde, sigma, Utilde, Vtilde, r); // i = 1, ..., r
        // SVD<SVDMethod::Jacobi> svd(Ctilde);
        perform_SVD(Ctilde, Utilde, sigma, Vtilde, r, svd_type);
    
        std::cout << "Check dimensions of Utilde: " << Utilde.rows() << " * " << Utilde.cols() << std::endl;
        std::cout << "Check dimensions of sigma:  " << sigma.size() << std::endl;
        std::cout << "Check dimensions of Vtilde: " << Vtilde.rows() << " * " << Vtilde.cols() << std::endl;

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
        // std::cout << "Check Xh: " << std::endl << Xh << std::endl << std::endl;
        // std::cout << "The square root of Xh is: " << std::endl << Xh_sqrt << std::endl << std::endl;
        // std::cout << "Check if Xh = Xh_sqrt * Xh_sqrt: " << std::endl << Xh_sqrt*Xh_sqrt << std::endl << std::endl;
        std::cout << "Norm of ||Xh - Xh_sqrt * Xh_sqrt||: " << (Xh-Xh_sqrt*Xh_sqrt).norm() << std::endl << std::endl;

        Mat_m Ktilde = (((Xh_sqrt*S) * D) * S.transpose()) * Xh_sqrt;
        // Ktilde = (Nh*Nh) * (Nh*ns) * (ns*ns) * (ns*Nh) * (Nh*Nh) = Nh*Nh
        std::cout << "Case ns = " << ns << " > " << Nh << " = Nh" << std::endl;
        std::cout << "Check dimensions of Ktilde: " << Ktilde.rows() << " * " << Ktilde.cols() << std::endl;

        // Solve the eigenvalue problem through SVD
        // – Initialize the inputs required by the SVD method
        // – Note that the SVD method returns sigma as a vector, then it has to be converted into a diagonal matrix
        Mat_m Utilde = Mat_m::Zero(Nh, r);
        Mat_m Vtilde = Mat_m::Zero(Nh, r);

        // SVD(Ktilde, sigma, Utilde, Vtilde, r); // i = 1, ..., r
        // SVD<SVDMethod::Jacobi> svd(Ktilde);
        perform_SVD(Ktilde, Utilde, sigma, Vtilde, r, svd_type);

        std::cout << "Check dimensions of Utilde: " << Utilde.rows() << " * " << Utilde.cols() << std::endl;
        std::cout << "Check dimensions of sigma:  " << sigma.size() << std::endl;
        std::cout << "Check dimensions of Vtilde: " << Vtilde.rows() << " * " << Vtilde.cols() << std::endl;

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

    std::cout << "Check dimensions of W:      " << W.rows() << " * " << W.cols() << std::endl;
    // std::cout << "Check intermediate W in which all POD modes are stored: " << std::endl << W << std::endl << std::endl;

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
    std::cout << "Check dimensions of W:      " << W.rows() << " * " << W.cols() << std::endl;
    // std::cout << "Check final W in which the POD basis is stored: " << std::endl << W << std::endl << std::endl;

    return std::make_tuple(W, sigma);
}

// Standard incremental SVD for building POD – Algorithm 1
void POD::standard_iSVD(Mat_m &U, Mat_m &Sigma, Mat_m &V, const Vec_v c, const double tol, const double tol_sv)
{
    std::cout << "===================================================================" << std::endl;
    std::cout << "Standard incremental SVD" << std::endl << std::endl;
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
    std::cout << "Check Sigma:" << std::endl << Sigma << std::endl << std::endl;
    std::cout << "Check d:" << std::endl << d << std::endl << std::endl;
    std::cout << "Check p:" << std::endl << p << std::endl << std::endl;
    std::cout << "Check Q:" << std::endl << Q << std::endl << std::endl;

    // Step 2: SVD solution
    // – Initialize the inputs required by the SVD method
    // – Note that the SVD method returns sigmaQ as a vector, then it has to be converted into a diagonal matrix
    Vec_v sigmaQ = Vec_v::Zero(k+1);
    Mat_m SigmaQ = Mat_m::Zero(k+1, k+1);
    Mat_m UQ = Mat_m::Zero(k+1, k+1);
    Mat_m VQ = Mat_m::Zero(k+1, k+1);

    // SVD(Q, sigmaQ, UQ, VQ, k+1);
    // SVD<SVDMethod::Jacobi> svd(Q); // (C, sigma, U, V, dim);
    perform_SVD(Q, UQ, sigmaQ, VQ, 0, 0); // poi aggiungi svd_type
 
    SigmaQ = sigmaQ.asDiagonal();
    std::cout << "Check UQ:" << std::endl << UQ << std::endl << std::endl;
    std::cout << "Check SigmaQ:" << std::endl << SigmaQ << std::endl << std::endl;
    std::cout << "Check VQ:" << std::endl << VQ << std::endl << std::endl;

    std::cout << "Compare intermediate and final dimensions with the initial k = " << k << std::endl << std::endl;

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

    std::cout << "Intermediate dimensions: ";
    if ((p < tol || k >= n) == true)
        std::cout << "in step 3 the 'if' branch is taken" << std::endl;
    else
        std::cout << "in step 3 the 'else' branch is taken" << std::endl;
    std::cout << "Check dimensions of U:     " << U.rows() << " * " << U.cols() << std::endl;
    std::cout << "Check dimensions of Sigma: " << Sigma.rows() << " * " << Sigma.cols() << std::endl;
    std::cout << "Check dimensions of V:     " << V.rows() << " * " << V.cols() << std::endl << std::endl;

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

    std::cout << "Final dimensions: " << std::endl;
    if (((Sigma(k-2, k-2) > tol_sv) && (Sigma(k-1, k-1) < tol_sv)) == true)
        std::cout << "in step 5 the 'if' branch is taken" << std::endl;
    else
        std::cout << "in step 5 the 'if' branch is not taken" << std::endl;
    std::cout << "Check dimensions of U:     " << U.rows() << " * " << U.cols() << std::endl;
    std::cout << "Check dimensions of Sigma: " << Sigma.rows() << " * " << Sigma.cols() << std::endl;
    std::cout << "Check dimensions of V:     " << V.rows() << " * " << V.cols() << std::endl << std::endl;


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
    std::cout << "===================================================================" << std::endl;
    std::cout << "Enhanced incremental SVD" << std::endl << std::endl;
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
    std::cout << "Check Sigma:" << std::endl << Sigma << std::endl << std::endl;
    std::cout << "Check d:" << std::endl << d << std::endl << std::endl;
    std::cout << "Check p:" << std::endl << p << std::endl << std::endl;
    std::cout << "Check Q:" << std::endl << Q << std::endl << std::endl;

    // Step 2: SVD solution
    // – Initialize the inputs required by the SVD method
    // – Note that the SVD method returns sigmaQ as a vector, then it has to be converted into a diagonal matrix
    Vec_v sigmaQ = Vec_v::Zero(k+1);
    Mat_m SigmaQ = Mat_m::Zero(k+1, k+1);
    Mat_m UQ = Mat_m::Zero(k+1, k+1);
    Mat_m VQ = Mat_m::Zero(k+1, k+1);

    // SVD(Q, sigmaQ, UQ, VQ, k+1);
    // SVD<SVDMethod::Jacobi> svd(Q); // (C, sigma, U, V, dim);
    perform_SVD(Q, UQ, sigmaQ, VQ, 0, 0); // poi aggiungi svd_type

    SigmaQ = sigmaQ.asDiagonal();
    std::cout << "Check UQ:" << std::endl << UQ << std::endl << std::endl;
    std::cout << "Check SigmaQ:" << std::endl << SigmaQ << std::endl << std::endl;
    std::cout << "Check VQ:" << std::endl << VQ << std::endl << std::endl;

    std::cout << "Compare intermediate and final dimensions with the initial k = " << k << std::endl << std::endl;

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

    std::cout << "Intermediate dimensions: " << std::endl;
    if ((p < tol || k >= n) == true)
        std::cout << "in step 3 the 'if' branch is taken" << std::endl;
    else
        std::cout << "in step 3 the 'else' branch is taken" << std::endl;
    std::cout << "Check dimensions of U:     " << U.rows() << " * " << U.cols() << std::endl;
    std::cout << "Check dimensions of Sigma: " << Sigma.rows() << " * " << Sigma.cols() << std::endl;
    std::cout << "Check dimensions of V:     " << V.rows() << " * " << V.cols() << std::endl << std::endl;

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

    std::cout << "Final dimensions: " << std::endl;
    if (k == M)
        std::cout << "in step 4 the 'if' branch is taken, " << std::endl;
    else
        std::cout << "in step 4 the 'if' branch is not taken, " << std::endl;
    if (((Sigma(k-2, k-2) > tol_sv) && (Sigma(k-1, k-1) < tol_sv)) == true)
        std::cout << "in step 5 the 'if' branch is taken" << std::endl;
    else
        std::cout << "in step 5 the 'if' branch is not taken" << std::endl;
    std::cout << "Check dimensions of U:     " << U.rows() << " * " << U.cols() << std::endl;
    std::cout << "Check dimensions of Sigma: " << Sigma.rows() << " * " << Sigma.cols() << std::endl;
    std::cout << "Check dimensions of V:     " << V.rows() << " * " << V.cols() << std::endl << std::endl;







    // FUNZIONA TOGLIENDO DIRETTAMENTE?


    // FARE PER ROBUSTEZZA O EVITARE?

    // Step 6: reorthogonalization
    // Note that epsilon is a double-precision machine epsilon
    // if ((V.col(k-1).transpose()*V.col(0)).norm() > std::min(tol, n*std::numeric_limits<double>::epsilon())) {
    //     V = modifiedGramSchmidt(V);
    // }
}

