import os
import numpy as np
from scipy.io import mmread
from scipy.linalg import qr
from sklearn.utils.extmath import randomized_svd

input_directory = "../data/input/"
output_directory = "../data/output/SVD/py"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

def save_vector_market(vector, output_path):
    with open(output_path, 'w') as file:
        length = len(vector)
        total_elements = length

        file.write("%%MatrixMarket matrix array real general\n")
        file.write(f"{length} 1\n")

        for i in range(length):
            value = vector[i]
            file.write(f"{i+1} {1} {value:.18e}\n")


def save_matrix_market(matrix, output_path):
    with open(output_path, 'w') as file:
        rows, cols = matrix.shape
        total_elements = matrix.size

        file.write("%%MatrixMarket matrix coordinate real general\n")
        file.write(f"{rows} {cols} {total_elements}\n")

        for j in range(cols):
            for i in range(rows):
                value = matrix[i, j]
                file.write(f"{i+1} {j+1} {value:.18e}\n")

    

def process_matrix(matrix_path):
    # Read the matrix from the Matrix Market file
    sparse_matrix = mmread(matrix_path)
    dense_matrix = sparse_matrix.toarray()

    # Perform SVD
    U, S, VT = np.linalg.svd(dense_matrix)
    
    matrix_name = os.path.splitext(os.path.basename(matrix_path))[0]

    
    U_output_path = os.path.join(output_directory, f"{matrix_name}_U.mtx")
    save_matrix_market(U, U_output_path)
    S_output_path = os.path.join(output_directory, f"{matrix_name}_S.mtx")
    save_matrix_market(np.diag(S), S_output_path)
    V_output_path = os.path.join(output_directory, f"{matrix_name}_V.mtx")
    save_matrix_market(VT, V_output_path)


def main():
    # Get a list of all .mtx files in the input directory
    mtx_files = [file for file in os.listdir(input_directory) if file.endswith(".mtx")]

    if not mtx_files:
        print("No .mtx files found in the input directory.")
        return

    for mtx_file in mtx_files:
        mtx_path = os.path.join(input_directory, mtx_file)
        # print(f"Processing {mtx_path}...")

        try:
            process_matrix(mtx_path)
            # print("rSVD completed and results saved.")
        except Exception as e:
            print(f"Error processing {mtx_path}: {e}")

if __name__ == "__main__":
    main()
