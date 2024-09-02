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

rom_sizes = list(map(int, parameters['rom_sizes'].split()))


# n = int(sys.argv[4])
# rom_sizes = list(map(int, sys.argv[5:]))

# Plot the solution
plt.rcParams.update({"font.size": 8})
# colors = plt.cm.viridis(np.linspace(0, 1, len(rom_sizes)))
# colors = plt.cm.jet(np.linspace(0, 1, len(rom_sizes)))
colors = plt.cm.tab20(np.linspace(0, 1, len(rom_sizes)))

for i in range(n):
    plt.plot(full_data[:,i], label = "FOM", color='black')
    for j in range(len(rom_sizes)):
        plt.plot(reconstructed_data[:,j*n+i], 'o', markerfacecolor='None', markersize=2, label = f"{rom_sizes[j]} POD modes", color=colors[j])
        plt.xlabel("x")
        plt.ylabel("u")
        mu_print = f"{mu[i]:.4f}"
        plt.title(f"Solution for mu = {mu_print}")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.subplots_adjust(right=0.8)
    plt.savefig(f"plot_{mu_print}mu.pdf", bbox_inches='tight')
    plt.close()

# Plot the error
for j in range(len(rom_sizes)):
    plt.plot(mu, errors_data[:,j], marker='o', markersize=2, label = f"{rom_sizes[j]} POD modes")
    plt.xlabel("mu")
    plt.xticks(mu)
    plt.ylabel("error")
    plt.yscale("log")
    plt.title(f"Relative errors for different numbers of POD modes")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.8)
plt.savefig(f"error.pdf", bbox_inches='tight')
plt.close()



# plt.rcParams.update({"font.size": 14})

# plt.plot(convergence_data.h,
#          convergence_data.eL2,
#          marker = 'o',
#          label = 'L2')
# plt.plot(convergence_data.h,
#          convergence_data.eH1,
#          marker = 'o',
#          label = 'H1')
# plt.plot(convergence_data.h,
#          convergence_data.h,
#          '--',
#          label = 'h')
# plt.plot(convergence_data.h,
#          convergence_data.h**2,
#          '--',
#          label = 'h^2')
# plt.plot(convergence_data.h,
#          convergence_data.h**3,
#          '--',
#          label = 'h^3')

# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("h")
# plt.ylabel("error")
# plt.legend()

# plt.savefig("convergence.pdf")