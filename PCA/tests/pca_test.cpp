#include "../include/PCA_class.hpp"
#include "../include/SVD_class.hpp"
#include <Eigen/Dense>
#include <iostream>

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

Mat loadTouristsData(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file");
    }

    std::string line;
    std::vector<std::vector<double>> data;

    // Read the header line and ignore
    std::getline(file, line);

    // Read the file line by line
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        std::string value;
        std::vector<double> row;

        // Skip the first three columns (categorical variables)
        std::getline(stream, value, ' '); // Skip first column
        std::getline(stream, value, ' '); // Skip second column
        std::getline(stream, value, ' '); // Skip third column

        // Read the remaining columns
        while (stream >> value) {
            try {
                row.push_back(std::stod(value));
            } catch (const std::invalid_argument&) {
                //std::cerr << "Skipping invalid value: " << value << " in row: " << line << std::endl;
            }
        }

        if (row.size() == 8) { // Ensure the row has the correct number of columns
            data.push_back(row);
        } 
    }

    file.close();

    // Convert the vector of vectors to Eigen::MatrixXd
    size_t rows = data.size();
    size_t cols = rows > 0 ? data[0].size() : 0;
    Mat matrix(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix(i, j) = data[i][j];
        }
    }

    return matrix;
}
Mat loadDataset(const std::string& filename) {
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
    Mat csv_to_mat(std::string & path){
  
    // Leggi il file CSV in una matrice Eigen
    std::vector<double> data;
    std::ifstream file(path);
    std::string line;
    int rows = 0;
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            data.push_back(std::stod(cell));
        }
        ++rows;
    }
    Eigen::MatrixXd A(rows, data.size()/rows);
int idx = 0;
for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < data.size()/rows; ++j) {
        A(i, j) = data[idx++];
    }
}
return A;
}

int main() {
    // Esempio di dati
    MPI_Init(NULL, NULL);
    std::string path = "ovariancancer_obs.csv";
    std::string path2 = "../data/input/dataset_athletic.txt";
    std::string path3 ="../data/input/tourists.txt";
    Mat data = loadTouristsData(path3);

        
        // Creazione dell'oggetto PCA
        PCA<SVDMethod::ParallelJacobi> pca(data, false);
//std::cout<<"S :"<<pca.getS()<<std::endl;   
        // Test dei metodi principali
        //std::cout << "Mean: " << pca.mean().transpose() << std::endl;
        /*std::cout << "Principal directions: " << std::endl << pca.principalDirections() << std::endl;
        std::cout << "Scores: " << std::endl << pca.scores() << std::endl;
        std::cout << "Explained Variance: " << pca.explainedVariance().transpose() << std::endl;
        std::cout << "Explained Variance Ratio: " << pca.explainedVarianceRatio().transpose() << std::endl;
*/
        // Test di salvataggio e caricamento
        /*pca.save("pca_state.txt");
        PCA<SVDMethod::SomeMethod> pca_loaded(data, true);
        pca_loaded.load("pca_state.txt");*/

        // Verifica dell'ortogonalità
        pca.summary();

        std::cout << "Orthogonality Check: " << pca.checkOrthogonality() << std::endl;
    //pca.saveResults("../data/output/pca_tourists_results.txt");
    MPI_Finalize();
    return 0;
}

