
#include "JacobiOperations.hpp"
#include "Jacobi_Class.hpp"


void applyOnTheLeft(Mat_m &matrix, int p, int q, double c, double s) {
    //apply_rotation_in_the_plane(matrix.row(p), matrix.row(q), matrix.cols(), j.getS(), j.getC());
    for (int i = 0; i < matrix.cols(); ++i) {
        double xi = matrix(p,i);
        double yi = matrix(q,i);
        matrix(p,i) = c * xi + (s) * yi;
        matrix(q,i) = -s * xi + (c) * yi;
}
}

void applyOnTheRight(Mat_m &matrix, int p, int q, double c, double s) {

    for (int i = 0; i < matrix.rows(); ++i) {
        double xi = matrix(i,p);
        double yi = matrix(i,q);
        matrix(i,p) = c * xi + (-s) * yi;
        matrix(i,q) = s * xi + (c) * yi;
    }
}
void real_2x2_jacobi_svd(Mat_m &matrix, double &c_left,double &s_left,double &c_right,double &s_right,int p, int q) {
  
    Mat_m m=Eigen::MatrixXd::Zero(2,2);
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
    
    
    Mat_m j_right(2,2);
    j_right<<c_right,s_right,
            -s_right,c_right;
    Mat_m rot_to_eigen(2,2);
    rot_to_eigen<<rot1.getC(),rot1.getS(),
                -rot1.getS(),rot1.getC();
                
    Mat_m left(2,2);
    left= rot_to_eigen * j_right.transpose();
    
    c_left=left(0,0);
    s_left=left(0,1);
    

    
}
bool svd_precondition_2x2_block_to_be_real(Mat_m& m_workMatrix, int p, int q, double maxDiagEntry) {
    // Estrai il blocco 2x2 dalla matrice di lavoro
    Mat_m block=Eigen::MatrixXd::Zero(2,2);
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

std::vector<std::pair<size_t, size_t>> greedy_maximum_weight_matching(const std::vector<std::tuple<double, size_t, size_t>>& weights) {
    std::vector<std::pair<size_t, size_t>> matching;
    std::vector<bool> used(weights.size(), false);

    for (const auto& [weight, i, j] : weights) {
        if (!used[i] && !used[j]) {
            matching.emplace_back(i, j);
            used[i] = true;
            used[j] = true;
        }
    }
    return matching;
}
// for the parallel version

void applyOnTheLeft_par(Mat_m &matrix, size_t p, size_t q, double c, double s) {
    #pragma omp parallel for
    for (Eigen::Index i = 0; i < matrix.cols(); ++i) {
        double xi = matrix(p,i);
        double yi = matrix(q,i);
        matrix(p,i) = c * xi + s * yi;
        matrix(q,i) = -s * xi + c * yi;
    }
}

void applyOnTheRight_par(Mat_m &matrix, size_t p, size_t q, double c, double s) {
    #pragma omp parallel for
    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
        double xi = matrix(i,p);
        double yi = matrix(i,q);
        matrix(i,p) = c * xi - s * yi;
        matrix(i,q) = s * xi + c * yi;
    }
}

void real_2x2_jacobi_svd_par(Mat_m &matrix, double &c_left,double &s_left,double &c_right,double &s_right,size_t  p, size_t  q) {
  
    Mat_m m=Eigen::MatrixXd::Zero(2,2);
   m << matrix(p,p), matrix(p,q),
        matrix(q,p), matrix(q,q);
        
    double t = m(0,0) + m(1,1);
    double d = m(1,0) - m(0,1);
    
    JacobiRotation rot1;
    if(d == 0) {
        rot1.setS(0.0);
        rot1.setC(1.0);
        
    } else {
      
        double u = t / d;
        double tmp = sqrt(1.0 + u*u);
        
        rot1.setS(1 / tmp);
        rot1.setC(u / tmp);
        
    }
    size_t a1=0;
    size_t a2=1;
    applyOnTheLeft_par(m,a1,a2,rot1.getC(),rot1.getS());
    
    double deno = 2 * std::abs(m(0,1));
    if (deno < 1e-10/*std::numeric_limits<double>::min()*/) {
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
    
    Mat_m j_right(2,2);
    j_right<<c_right,s_right,
            -s_right,c_right;
    Mat_m rot_to_eigen(2,2);
    rot_to_eigen<<rot1.getC(),rot1.getS(),
                -rot1.getS(),rot1.getC();
                
    Mat_m left(2,2);
    left= rot_to_eigen * j_right.transpose();
    
    c_left=left(0,0);
    s_left=left(0,1);
    

    
}

