import os
import numpy as np
from scipy.io import mmread
from scipy.linalg import qr

input_directory = "../data/input/"
output_directory = "../data/output/QR/py"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

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

    # Perform QR decomposition
    Q, R = np.linalg.qr(dense_matrix, mode="reduced")

    # Save Q matrix in Matrix Market format
    matrix_name = os.path.splitext(os.path.basename(matrix_path))[0]
    Q_output_path = os.path.join(output_directory, f"{matrix_name}_Q.mtx")
    save_matrix_market(Q, Q_output_path)

    # Save R matrix in Matrix Market format
    R_output_path = os.path.join(output_directory, f"{matrix_name}_R.mtx")
    save_matrix_market(R, R_output_path)

def main():
    # Get a list of all .mtx files in the input directory
    mtx_files = [file for file in os.listdir(input_directory) if file.endswith(".mtx")]

    if not mtx_files:
        print("No .mtx files found in the input directory.")
        return

    for mtx_file in mtx_files:
        mtx_path = os.path.join(input_directory, mtx_file)
        print(f"Processing {mtx_path}...")

        try:
            process_matrix(mtx_path)
            print("QR decomposition completed and results saved.")
        except Exception as e:
            print(f"Error processing {mtx_path}: {e}")

if __name__ == "__main__":
    main()
