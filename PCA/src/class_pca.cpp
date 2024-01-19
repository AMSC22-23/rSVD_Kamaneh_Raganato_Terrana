#include "class_pca.hpp"

    
    void PCA::setEigenvalues(const Vec& values) {
    eigenvalues = values;
}

void PCA::setEigenvectors(const Mat& vectors) {
    eigenvectors = vectors;
}
    

    void PCA::compute() {
        
        // Center the data
        mean_center(data_);
        
        if (use_svd_) {
       
        // Set the centered data as the data for the SVD
        setData(data_);

        // Compute the SVD of the centered data
        
        SVD::compute();

        // The eigenvalues are the singular values squared
        U_ = getU();
        S_ = getS();
        V_ = getV();
        


    } else {
        
        Mat cov = computeCovarianceMatrix();

        // Compute the EVD of the covariance matrix
        Eigen::SelfAdjointEigenSolver<Mat> es(cov);

        // The eigenvalues are the eigenvalues of the covariance matrix
        setEigenvalues(es.eigenvalues());

        // The eigenvectors are the eigenvectors of the covariance matrix
        setEigenvectors(es.eigenvectors());
    }
    }
    Mat PCA::scores() {
    // Ensure that PCA has been computed
    if(scores_.size() == 0) {
        
        if(use_svd_){
            if (S_.size() == 0) {
            compute();
    }
     scores_ = U_ * S_.asDiagonal();
    
    }
    else{
        if (eigenvectors.size() == 0) {
        compute();
    }
    
    // Calculate the principal components
     scores_ = data_ * eigenvectors;
    }
    }

    return scores_;
}

    
    
    Mat PCA::computeCovarianceMatrix() {
        int n = data_.rows();
        Mat cov= Mat::Zero(data_.cols(),data_.cols());
        cov =data_.transpose()*data_;
    return cov/(n-1);
    }

   /* Mat getEigenvectors() const {
        return eigenvectors;
    }

    Vec getEigenvalues() const {
        return eigenvalues;
    }*/


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

void PCA::mean_center( Mat& data){
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

/*void PCA::scree_plot(const std::string& filename) const {
    // This method generates a scree plot, which is a plot of the explained variance ratio
    // This requires a plotting library, such as gnuplot or ROOT
    // The implementation will depend on the specific library you are using
    // Here is a pseudocode example:
    /*
    Plot plot(filename);
    plot.add(explained_variance_ratio());
    plot.xlabel("Principal Component");
    plot.ylabel("Explained Variance Ratio");
    plot.save();
    
}*/







/*void PCA::fit(const Mat& data, bool use_svd) {
    if (use_svd) {
        // Center the data
        Mat centered_data = data.rowwise() - data.colwise().mean();

        // Compute the SVD of the centered data
        SVD::compute(centered_data);

        // The eigenvalues are the singular values squared
        eigenvalues = getS().array().square();

        // The eigenvectors are the right singular vectors
        eigenvectors = getV();
    } else {
        // Compute the covariance matrix
        Mat cov = computeCovarianceMatrix(data);

        // Compute the EVD of the covariance matrix
        Eigen::SelfAdjointEigenSolver<Mat> es(cov);

        // The eigenvalues are the eigenvalues of the covariance matrix
        eigenvalues = es.eigenvalues();

        // The eigenvectors are the eigenvectors of the covariance matrix
        eigenvectors = es.eigenvectors();
    }
}*/