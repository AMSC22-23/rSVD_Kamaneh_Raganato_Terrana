#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Read the csv file containing the parameters and the errors
data = pd.read_csv(sys.argv[1], sep = ",")
df = pd.DataFrame(data, columns=['N', 'beta', 'amplitude', 'rom_size', 'T', 'deltat', 'error1', 'error2', 'error3', 'error4', 'error5'])

# Group by the first three columns
grouped = df.groupby(['N'])
# , 'beta', 'amplitude'])

# Plot the errors
for i in range(1, 5):
    plt.figure(figsize=(15, 10))
    for name, group in grouped:
        errors = group.iloc[:, 6 + i].values.flatten()
        x = np.arange(len(errors))
        
        # Plot each group with a unique label
        plt.plot(x, errors, label=f'Group {name}')
    
    plt.xlabel('Error Index')
    plt.ylabel('Error Value')
    plt.title(f'Error Plot Grouped by N, beta, amplitude for error{i}')
    plt.legend()
    plt.savefig(f"errors_n1_{i}.pdf", bbox_inches='tight')
    plt.show()

# # convert in numpy array
# df = df.to_numpy()

# # Group by the first three columns
# grouped = df.groupby(['N', 'beta', 'amplitude'])

# #plot the errors
# for i in range(1, 6):
    
#         plt.figure(figsize=(15, 10))
#         for name, group in grouped:
#             errors = group.iloc[:, 6+i].values.flatten()
            
#             x = np.arange(len(errors))
            
#             # Plot each group with a unique label
#             plt.plot(x, errors, label=f'Group {name}')
    
#         plt.xlabel('Error Index')
#         plt.ylabel('Error Value')
#         plt.title(f'Error Plot Grouped by col0, col1, col2 for error{i}')
#         plt.savefig(f"errors_n1_{i}.pdf", bbox_inches='tight')
#         # plt.legend()
#         plt.show()

# for i in range(1, 6):

#     plt.figure(figsize=(15, 10))
#     grouped = df.groupby(['N', 'beta', 'amplitude'])
#     for name, group in grouped:
#         errors = group.iloc[:, 6+i].values.flatten()
        
#         x = np.arange(len(errors))
        
#         # Plot each group with a unique label
#         plt.plot(x, errors, label=f'Group {name}')

#     plt.xlabel('Error Index')
#     plt.ylabel('Error Value')
#     plt.title(f'Error Plot Grouped by col0, col1, col2 for error{i}')
#     plt.savefig(f"errors_n1_{i}.pdf", bbox_inches='tight')
#     # plt.legend()
#     plt.show()

# grouped = df.groupby(['N', 'beta', 'amplitude'])

# # Plot the errors
# plt.figure(figsize=(15, 10))
# for name, group in grouped:
#     errors = group.iloc[:, 6:].values.flatten()
    
#     x = np.arange(len(errors))
    
#     # Plot each group with a unique label
#     plt.plot(x, errors, label=f'Group {name}')

# plt.xlabel('Error Index')
# plt.ylabel('Error Value')
# plt.title('Error Plot Grouped by col0, col1, col2')
# plt.savefig(f"errors_n1.pdf", bbox_inches='tight')
# # plt.legend()
# plt.show()


# Group by the first four columns
# grouped = data.groupby([0, 1, 2, 3])

# # Iterate over each group and plot the errors
# for name, group in grouped:
#     errors = group.iloc[:, 6:]  # Select the error columns
#     plt.figure()
#     plt.plot(errors.T)
#     plt.title(f'Group: {name}')
#     plt.xlabel('Error Index')
#     plt.ylabel('Error Value')
#     plt.savefig(f"errors_n1.pdf", bbox_inches='tight')

# # Read the txt file containing the parameters
# parameter_file = sys.argv[4]
# parameters = {}
# with open(parameter_file, 'r') as file:
#     for line in file:
#         if line.startswith('#') or not line.strip():
#             continue
#         key, value = line.split(maxsplit=1)
#         value = value.split('#')[0].strip()
#         parameters[key] = value.strip()
# n = int(parameters['n'])
# mu_min = float(parameters['mu_min'])
# mu_max = float(parameters['mu_max'])
# mu = np.linspace(mu_min, mu_max, n)

# rom_sizes = list(map(int, parameters['rom_sizes'].split()))


# # n = int(sys.argv[4])
# # rom_sizes = list(map(int, sys.argv[5:]))

# # Plot the solution
# plt.rcParams.update({"font.size": 8})
# colors = plt.cm.tab10(np.linspace(0, 1, len(rom_sizes)))

# for i in range(n):
#     plt.plot(full_data[:,i], label = "FOM", color='gray')
#     for j in range(len(rom_sizes)):
#         plt.plot(reconstructed_data[:,j*n+i], 'o', markerfacecolor='None', markersize=4, label = f"{rom_sizes[j]} POD modes", color=colors[j])
#         plt.xlabel("x")
#         plt.ylabel("u")
#         mu_print = f"{mu[i]:.4f}"
#         plt.title(f"Solution for mu = {mu_print}")
#         plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#         plt.subplots_adjust(right=0.8)
#     plt.savefig(f"plot_{mu_print}mu_n{n}.pdf", bbox_inches='tight')
#     plt.close()

# # Plot the error
# for j in range(len(rom_sizes)):
#     plt.plot(mu, errors_data[:,j], marker='o', markersize=2, label = f"{rom_sizes[j]} POD modes", color=colors[j])
#     plt.xlabel("mu")
#     plt.xticks(mu)
#     plt.ylabel("error")
#     plt.yscale("log")
#     plt.title(f"Relative errors for different numbers of POD modes")
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#     plt.subplots_adjust(right=0.8)
# plt.savefig(f"error.pdf", bbox_inches='tight')
# plt.close()

