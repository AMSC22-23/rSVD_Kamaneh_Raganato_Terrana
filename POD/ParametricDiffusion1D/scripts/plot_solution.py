#!/usr/bin/env python
from scipy.io import mmread
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Read the matrices containing the full and reconstructed solutions for all the parameters and rom sizes
full_data = mmread(sys.argv[1])
reconstructed_data = mmread(sys.argv[2])

full_data = full_data.todense()
reconstructed_data = reconstructed_data.todense()

# Read the matrix containing the errors for all the parameters and rom sizes
errors_data = mmread(sys.argv[3])
errors_data = errors_data.todense()

# Read the txt file containing the parameters
parameter_file = sys.argv[4]
parameters = {}
with open(parameter_file, 'r') as file:
    for line in file:
        if line.startswith('#') or not line.strip():
            continue
        key, value = line.split(maxsplit=1)
        value = value.split('#')[0].strip()
        parameters[key] = value.strip()
n = int(parameters['n'])
mu_min = float(parameters['mu_min'])
mu_max = float(parameters['mu_max'])
mu = np.linspace(mu_min, mu_max, n)
if parameters['mu_new']:
    mu_new = float(parameters['mu_new'])

rom_sizes = list(map(int, parameters['rom_sizes'].split()))


# n = int(sys.argv[4])
# rom_sizes = list(map(int, sys.argv[5:]))

# Plot the solution
plt.rcParams.update({"font.size": 8})
colors = plt.cm.tab10(np.linspace(0, 1, len(rom_sizes)))

for i in range(full_data.shape[1]):
    plt.plot(full_data[:,i], label = "FOM", color='gray')
    for j in range(len(rom_sizes)):
        plt.plot(reconstructed_data[:,j*full_data.shape[1]+i], 'o', markerfacecolor='None', markersize=4, label = f"{rom_sizes[j]} POD modes", color=colors[j])
        plt.xlabel("x")
        plt.ylabel("u")
        if mu_new:
            mu_print = mu_new
        else:
            mu_print = f"{mu[i]:.4f}"
        plt.title(f"Solution for mu = {mu_print}")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.subplots_adjust(right=0.8)
    plt.savefig(f"plot_{mu_print}mu_n{full_data.shape[1]}.pdf", bbox_inches='tight')
    plt.close()

tick_lab = np.linspace(mu_min, mu_max, 5)

# Plot the error
if not mu_new:
    for j in range(len(rom_sizes)):
        plt.plot(mu, errors_data[:,j], marker='o', markersize=2, label = f"{rom_sizes[j]} POD modes", color=colors[j])
        plt.xlabel("mu")
        plt.xticks(tick_lab)
        plt.ylabel("error")
        plt.yscale("log")
        plt.title(f"Relative errors for different numbers of POD modes")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.subplots_adjust(right=0.8)
    plt.savefig(f"error.pdf", bbox_inches='tight')
    plt.close()

if mu_new:
    for j in range(len(rom_sizes)):
        plt.plot(mu_new, errors_data[:,j], marker='o', markersize=2, label = f"{rom_sizes[j]} POD modes", color=colors[j])
        plt.xlabel("mu_new")
        plt.xticks(tick_lab)
        plt.ylabel("error")
        plt.yscale("log")
        plt.title(f"Relative errors for different numbers of POD modes")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.subplots_adjust(right=0.8)
    plt.savefig(f"error.pdf", bbox_inches='tight')
    plt.close()

