#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_dataset>" << std::endl;
        return 1;
    }

    std::string dataset_path = argv[1];
    std::cout << "Running PCA on the dataset: " << dataset_path << std::endl;

    PCA<SVDMethod::ParallelJacobi> pca(csv_to_mat(dataset_path), false);

    return 0;
}