# Try importing PyQt5
import PyQt5

import matplotlib
#matplotlib.use('Qt5Agg')  # Set the backend to Qt5Agg
import pandas as pd
import matplotlib.pyplot as plt

import os

def read_pca_results(filename):
    """
    Reads PCA results from a specified file and extracts explained variance ratios
    and scores. Assumes a specific format for sections and content.

    Parameters:
    filename (str): The path to the file containing PCA results.

    Returns:
    tuple: A tuple containing two lists, one for the variance ratios and another for the scores.
    """
    # Ensure the file exists to avoid FileNotFoundError
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such file: {filename}")

    with open(filename, 'r') as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]

    try:
        # Find the starting index for each relevant section
        idx_var_ratio = lines.index("Explained Variance Ratio:")
        idx_scores = lines.index("Scores:")
    except ValueError as e:
        print(f"Error: {str(e)} - Header not found in the file.")
        exit(1)  # Exit if headers are not found to avoid running with incorrect data

    # Parsing explained variance ratios from the file
    variance_ratios = [float(x) for x in lines[idx_var_ratio + 1:idx_scores]]

    # Parsing scores, split by commas and convert each to float
    scores = [list(map(float, line.split(', '))) for line in lines[idx_scores + 1:]]

    return variance_ratios, scores

def plot_variance_ratios(variance_ratios):
    """
    Plots the explained variance ratios using matplotlib.

    Parameters:
    variance_ratios (list): A list of variance ratios.
    """
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(variance_ratios)), variance_ratios, alpha=0.7)
    plt.title('Explained Variance Ratios')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Ratio')
    #plt.show()

def plot_scores(scores):
    """
    Plots the principal component scores as a scatter matrix using pandas and matplotlib.

    Parameters:
    scores (list of list): A nested list where each sublist represents scores for a principal component.
    """
    df_scores = pd.DataFrame(scores, columns=[f'PC{i+1}' for i in range(len(scores[0]))])
    plt.figure(figsize=(8, 8))
    pd.plotting.scatter_matrix(df_scores, alpha=0.8, diagonal='kde')
    plt.suptitle('Scatter Matrix of Principal Component Scores')
    plt.show()

if __name__ == "__main__":
    try:
        # Assuming the results file is named 'pca_results.txt'
        variance_ratios, scores = read_pca_results('../data/output/pca_tourists_results.txt')
        plot_variance_ratios(variance_ratios)
        plot_scores(scores)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
