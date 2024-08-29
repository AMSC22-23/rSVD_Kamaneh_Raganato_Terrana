import scipy.sparse as sp
import numpy as np

def save_sparse_matrix_market(matrix, output_path):
    rows, cols = matrix.shape
    total_elements = matrix.nnz

    with open(output_path, 'w') as file:
        file.write("%%MatrixMarket matrix coordinate real general\n")
        file.write(f"{rows} {cols} {total_elements}\n")

        for i, j, value in zip(matrix.row, matrix.col, matrix.data):
            file.write(f"{i+1} {j+1} {value:.18e}\n")

# Create a sparse matrix with values from 1 to 100
size = 100
values = np.arange(1, size*size + 1)
row_indices = np.repeat(np.arange(size), size)
col_indices = np.tile(np.arange(size), size)

sparse_matrix = sp.coo_matrix((values, (row_indices, col_indices)), shape=(size, size))

# Save the sparse matrix in Matrix Market format
output_path = 'sparse_matrix.mtx'
save_sparse_matrix_market(sparse_matrix, output_path)
