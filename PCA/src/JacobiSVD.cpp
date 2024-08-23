
#include "../include/JacobiSVD.hpp"
#include "../include/Jacobi_Class.hpp"


void applyOnTheLeft(Mat &matrix, int p, int q, double c, double s) {
    //apply_rotation_in_the_plane(matrix.row(p), matrix.row(q), matrix.cols(), j.getS(), j.getC());
    for (int i = 0; i < matrix.cols(); ++i) {
        double xi = matrix(p,i);
        double yi = matrix(q,i);
        matrix(p,i) = c * xi + (s) * yi;
        matrix(q,i) = -s * xi + (c) * yi;
}
}

void applyOnTheRight(Mat &matrix, int p, int q, double c, double s) {

    for (int i = 0; i < matrix.rows(); ++i) {
        double xi = matrix(i,p);
        double yi = matrix(i,q);
        matrix(i,p) = c * xi + (-s) * yi;
        matrix(i,q) = s * xi + (c) * yi;
    }
}
void real_2x2_jacobi_svd(Mat &matrix, double &c_left,double &s_left,double &c_right,double &s_right,int p, int q) {
  
    Mat m=Eigen::MatrixXd::Zero(2,2);
   m << matrix(p,p), matrix(p,q),
        matrix(q,p), matrix(q,q);
        
    double t = m(0,0) + m(1,1);
    double d = m(1,0) - m(0,1);
    
    JacobiRotation rot1;
    if(d == 0) {
        rot1.setS(0.0);
        rot1.setC(1.0);
        //std::cout<<"check inside real_2x2_jacobi_svd   3"<<std::endl;
    } else {
      //std::cout<<"check inside real_2x2_jacobi_svd   4"<<std::endl;
        double u = t / d;
        double tmp = sqrt(1.0 + u*u);
        
        rot1.setS(1 / tmp);
        rot1.setC(u / tmp);
        
    }
    
    applyOnTheLeft(m,0,1,rot1.getC(),rot1.getS());
    
    double deno = 2 * std::abs(m(0,1));
    if (deno < std::numeric_limits<double>::min()) {
        c_right=1;
        s_right=0;
        
    } else {
        double tau = (m(0,0) - m(1,1)) / deno;
        double w = std::sqrt(tau * tau + 1);
        double t2;
        if (tau > 0) {
            t2 = 1 / (tau + w);
        } else {
            t2 = 1 / (tau - w);
        }
        double segno = t2 > 0 ? 1 : -1;
        double n = 1 / std::sqrt(t2 * t2 + 1);
        s_right=-segno * (m(0,1) / std::abs(m(0,1))) * std::abs(t2) * n;
        c_right=n;
        
    }
    
    //j_right.makeJacobi(m(0,0), m(0,1), m(1,1));
    
    //std::cout<<"check inside real_2x2_jacobi_svd   8"<<std::endl;
    Mat j_right(2,2);
    j_right<<c_right,s_right,
            -s_right,c_right;
    Mat rot_to_eigen(2,2);
    rot_to_eigen<<rot1.getC(),rot1.getS(),
                -rot1.getS(),rot1.getC();
                //std::cout<<"rot to eigen"<<rot_to_eigen<<std::endl;
    Mat left(2,2);
    left= rot_to_eigen * j_right.transpose();
    
    c_left=left(0,0);
    s_left=left(0,1);
    

    //std::cout<<"check inside real_2x2_jacobi_svd   9"<<std::endl;
}
bool svd_precondition_2x2_block_to_be_real(Mat& m_workMatrix, int p, int q, double maxDiagEntry) {
    // Estrai il blocco 2x2 dalla matrice di lavoro
    Mat block=Eigen::MatrixXd::Zero(2,2);
   block << m_workMatrix(p,p), m_workMatrix(p,q),
        m_workMatrix(q,p), m_workMatrix(q,q);

    // Controlla se gli elementi fuori dalla diagonale del blocco sono vicini a zero
    if (std::abs(block(0, 1)) < maxDiagEntry * std::numeric_limits<double>::epsilon() &&
        std::abs(block(1, 0)) < maxDiagEntry * std::numeric_limits<double>::epsilon()) {
        // Il blocco è già diagonale
        return false;
    }
    // Il blocco non è diagonale
    return true;
}
void JacobiRotationSVD(Mat &A, Mat &m_matrixU, Mat &m_matrixV, Vec &Sigma) {
    int m = A.rows();
    int n = A.cols();

    // Step 1: B = A
    Mat m_workMatrix = A;
    
    // Step 2: U = I_mxn
    m_matrixU = Mat::Identity(m, m);

    // Step 3: V = I_nxn
    m_matrixV = Mat::Identity(n, n);
    //std::cout<<"B: "<<U<<std::endl;
    // Step 4: If m >= n, compute the QR factorization of B
    /*if (m > n) {
        HouseholderQR<MatrixXd> qr(B);
         B = qr.matrixQR().topLeftCorner(n, n).triangularView<Upper>();
    U =  qr.householderQ()*U;
    }*/
    //std::cout<<"B: "<<U<<std::endl;
    // Step 5: Set N^2 - ∑∑b_ij^2 = 0_i,j=n and first = true
    /*if(m_rows != m_cols)
{
    m_scaledMatrix = matrix / scale;
    m_qr_precond_morecols.run(*this, m_scaledMatrix);
    m_qr_precond_morerows.run(*this, m_scaledMatrix);
}
else
{
    m_workMatrix = matrix.block(0, 0, m_diagSize, m_diagSize) / scale;
    if(m_computeFullU) m_matrixU = Eigen::MatrixXd::Identity(m_rows, m_rows);
    if(m_computeThinU) m_matrixU = Eigen::MatrixXd::Identity(m_rows, m_diagSize);
    if(m_computeFullV) m_matrixV = Eigen::MatrixXd::Identity(m_cols, m_cols);
    if(m_computeThinV) m_matrixV = Eigen::MatrixXd::Identity(m_cols, m_diagSize);
}*/
    bool finished = false;
    const double considerAsZero = (std::numeric_limits<double>::min)();
    
    const double precision = 2.0 *  std::numeric_limits<double>::epsilon();
    double maxDiagEntry = m_workMatrix.cwiseAbs().diagonal().maxCoeff();
    double c_left = 0, s_left = 0, c_right = 0, s_right = 0;
   
    
  while (!finished) {
    finished = true;

        for (int p = 1; p < n; ++p) {
            for (int q = 0; q < p; ++q) {
            

        double threshold = std::max(considerAsZero, precision * maxDiagEntry);
        //std::cout<<threshold<<" "<<considerAsZero<<" "<<precision*maxDiagEntry<<std::endl;
        
        if (std::abs(m_workMatrix(p,q)) > threshold || std::abs(m_workMatrix(q,p)) > threshold) {
            finished = false;
            // perform SVD decomposition of 2x2 sub-matrix corresponding to indices p,q to make it diagonal
            // the complex to real operation returns true if the updated 2x2 block is not already diagonal
            if (svd_precondition_2x2_block_to_be_real(m_workMatrix, p, q, maxDiagEntry)) {
             
                real_2x2_jacobi_svd(m_workMatrix, c_left,s_left,c_right,s_right,p, q);
               
                applyOnTheLeft(m_workMatrix, p, q,c_left,s_left);
                
                applyOnTheRight(m_matrixU, p, q, c_left, -s_left);
                
                applyOnTheRight(m_workMatrix, p, q, c_right, s_right);
                
                applyOnTheRight(m_matrixV, p, q, c_right, s_right);
                

                // keep track of the largest diagonal coefficient
                maxDiagEntry = std::max(maxDiagEntry, std::max(std::abs(m_workMatrix(p,p)), std::abs(m_workMatrix(q,q))));
                
            }
        }
        
    }
}

  }
    // Step 7: The work matrix is now diagonal, so ensure it's positive so its diagonal entries are the singular values
    for (int i = 0; i < m_workMatrix.rows(); ++i) {
        double a = m_workMatrix(i, i);
        Sigma(i) = std::abs(a);
        if (a < 0) m_matrixU.col(i) = -m_matrixU.col(i);
    }
  }

