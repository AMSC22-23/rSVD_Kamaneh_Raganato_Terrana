import os
import matplotlib.pyplot as plt

def read_matrix_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    dimensions = tuple(map(int, lines[1].split()))
    matrix_data = [(int(line.split()[0]), int(line.split()[1]), float(line.split()[2])) for line in lines[2:]]

    return dimensions, matrix_data

def compare_matrices(matrix1_path, matrix2_path):
    matrix1_dimensions, matrix1_data = read_matrix_file(matrix1_path)
    matrix2_dimensions, matrix2_data = read_matrix_file(matrix2_path)

    # print(matrix1_path)
    # print(matrix1_path)
    # print(matrix1_dimensions)

    if matrix1_dimensions != matrix2_dimensions:
        print(f"Dimension mismatch between {matrix1_path} and {matrix2_path}.")
        return

    differences = []

    for cell1, cell2 in zip(matrix1_data, matrix2_data):
        if cell1[0:2] != cell2[0:2]:
            print(f"Cell index mismatch between {matrix1_path} and {matrix2_path}.")
            return

        value_difference = abs(abs(cell1[2]) - abs(cell2[2]))
        differences.append(value_difference)

    if differences:
        variance = sum(differences) / len(differences)

    dimension = matrix1_dimensions[0]
    return dimension, variance
    

def main():
    directory1 = "../data/output/rSVD/my/"
    directory2 = "../data/output/rSVD/py/"

    files1 = [file for file in os.listdir(directory1) if file.endswith(".mtx")]
    files2 = [file for file in os.listdir(directory2) if file.endswith(".mtx")]

    results = []

    for file1, file2 in zip(files1, files2):
        matrix1_path = os.path.join(directory1, file1)
        matrix2_path = os.path.join(directory2, file2)

        dimension, variance = compare_matrices(matrix1_path, matrix2_path)
        results.append((dimension, variance))
    
    sorted_results = sorted(results, key=lambda x: x[0])
    dimensions, variances = zip(*sorted_results)

    for item in sorted_results:
        print (item)

    # Plotting
    plt.plot(dimensions, variances, color='b')
    plt.title('Comparison of Results')
    plt.xlabel('Dimension')
    plt.ylabel('Norm')
    # plt.grid(True)
    plt.savefig('variance_plot.png')
    plt.show()

if __name__ == "__main__":
    main()
