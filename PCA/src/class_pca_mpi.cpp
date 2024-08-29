#include "../include/class_pca_mpi.hpp"

    
    void PCA::setEigenvalues(const Vec& values) {
    eigenvalues = values;
}

void PCA::setEigenvectors(const Mat& vectors) {
    eigenvectors = vectors;
}
Eigen::VectorXd PCA::getS() const {
    if(use_svd_){
        return S_;
    }
    return eigenvalues;
}

Eigen::MatrixXd PCA::getV() const {
    if(use_svd_){
        return V_;
    }
    return eigenvectors;
}
    

    void PCA::compute() {
        
        // Center the data
        mean_center(data_);
        
        if (use_svd_) {
       
        // Set the centered data as the data for the SVD
        setData(data_);
        std::cout<<"chiamo compute"<<std::endl;
        // Compute the SVD of the centered data
        
        SVD::compute();

        // The eigenvalues are the singular values squared
        U_ = SVD::getU();
        S_ = SVD::getS();
        V_ = SVD::getV();
        
        // Broadcast the results to all processes
        /*MPI_Bcast(U_.data(), U_.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(S_.data(), S_.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(V_.data(), V_.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);*/


    } else {
        
        Mat cov = computeCovarianceMatrix();
        //std::cout<<"cov: "<<cov<<std::endl;
        SVD::JacobiSVD(cov);
        
        // The eigenvalues are the eigenvalues of the covariance matrix
        eigenvalues= SVD::getS()*(data_.rows()-1);
        eigenvalues = eigenvalues.array().sqrt();
        
        // The eigenvectors are the eigenvectors of the covariance matrix
        eigenvectors = getV();
        
        
    }
    }
    // Parallel version of scores
    Mat PCA::scores() {
    // Ensure that PCA has been computed
    if (scores_.size() == 0) {
        if (use_svd_) {
            if (S_.size() == 0) {
                compute();
            }
            /*/ Calculate the scores in parallel
            int rows_per_process = data_.rows() / mpi_size_;
            Mat local_data = data_.block(mpi_rank_ * rows_per_process, 0, rows_per_process, data_.cols());
            Mat local_scores = local_data * eigenvectors;
            
            // Gather the scores
            Mat global_scores(data_.rows(), data_.cols());
            MPI_Allgather(local_scores.data(), local_scores.size(), MPI_DOUBLE, global_scores.data(), local_scores.size(), MPI_DOUBLE, MPI_COMM_WORLD);
            
            scores_ = global_scores;*/
            if (full_matrices_){
            Mat temp= Mat::Zero(data_.rows(), data_.cols());
            for(size_t i=0;i<S_.size();i++){
                temp(i,i)=S_(i);
            }
            scores_ = U_ * temp;
            }
            else{
            scores_ = U_ * S_.asDiagonal();
            }
        } else {
            if (eigenvectors.size() == 0) {
                compute();
            }

            // Calculate the principal components
            

            Mat local_scores = Mat::Zero(data_.rows() / mpi_size_, data_.cols());
            for (size_t i = mpi_rank_; i < data_.rows(); i += mpi_size_) {
                local_scores.row(i / mpi_size_) = data_.row(i) * eigenvectors;
            }

            // Gather the scores
            Mat global_scores(data_.rows(), data_.cols());
            MPI_Allgather(local_scores.data(), local_scores.size(), MPI_DOUBLE, global_scores.data(), local_scores.size(), MPI_DOUBLE, MPI_COMM_WORLD);

            scores_ = global_scores;
        }
    }

    return scores_;
}

   
    
    Mat PCA::computeCovarianceMatrix() {
    // Calculate the number of rows per process and the number of remaining rows
    int rows_per_process = data_.rows() / mpi_size_;
    int remaining_rows = data_.rows() % mpi_size_;

    // Calculate the start and end row for this process
    int start_row = mpi_rank_ * rows_per_process + std::min(mpi_rank_, remaining_rows);
    int end_row = start_row + rows_per_process + (mpi_rank_ < remaining_rows ? 1 : 0);

    // Get the part of the data for this process
    Mat local_data = data_.block(start_row, 0, end_row - start_row, data_.cols());

    // Each process computes the product for its part of the data
    Mat local_product = local_data.transpose() * local_data;

    // Reduce the products
    Mat global_product = Mat::Zero(data_.cols(), data_.cols());
    MPI_Allreduce(local_product.data(), global_product.data(), local_product.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return global_product / (data_.rows() - 1);
}

     Mat PCA::loadDataset(const std::string& filename) {
        std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    std::vector<double> data;
    std::string line;
    int rows = 0;
    while (std::getline(file, line)) {
        if (rows == 0) { // Skip header row
            ++rows;
            continue;
        }
        std::stringstream lineStream(line);
        std::string cell;
        int cols=0;
        while (std::getline(lineStream, cell, ' ')) {
            if(cols==0){
                cols++;
                continue;
            }
            if (!cell.empty() && cell != "\"\"") {
                data.push_back(std::stod(cell));
            }
        }
        ++rows;
    }

    int cols = data.size() / (rows - 1); // Subtract 1 because we skipped the header row
    Mat matrix(rows - 1, cols);
    for (int i = 0; i < rows - 1; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = data[i * cols + j];
        }
    }

    return matrix;
    }

// Parallel version of mean_center
void PCA::mean_center(Mat& data) {
   

    Vec mean(data.cols());
    for(size_t i=0;i<data.cols();i++){
        mean(i)=data.col(i).mean();
    }
    for(size_t i=0;i<data.cols();i++){
        for(size_t j=0;j<data.rows();j++){
        data(j,i)-=mean(i);   
        }
}
}


Vec PCA::explained_variance() const {
    // The explained variance is the eigenvalues
    if(use_svd_){
        return S_.array().square();
    }
    return eigenvalues;
}

Vec PCA::explained_variance_ratio() const {
    if(use_svd_){
        return S_.array().square()/S_.array().square().sum();
    }
    // The explained variance ratio is the eigenvalues divided by the sum of the eigenvalues
    return eigenvalues / eigenvalues.sum();
}


/*Mat PCA::get_loadings() const {
    // The loadings are the correlations between the original variables and the principal components
    // This assumes that the original data is stored in a member variable `data`
    Mat centered_data = data.rowwise() - data.colwise().mean();
    Mat standardized_data = centered_data.array().rowwise() / centered_data.array().colwise().square().sum().sqrt();
    return standardized_data.transpose() * get_components();
}*/
