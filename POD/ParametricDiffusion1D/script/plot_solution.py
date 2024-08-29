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

# Read the number of parameters and the rom sizes
n = int(sys.argv[3])
rom_sizes = list(map(int, sys.argv[4:]))

# Plot
plt.rcParams.update({"font.size": 14})

for i in range(n):
    plt.plot(full_data[:,i], label = "FOM")
    for j in range(len(rom_sizes)):
        plt.plot(reconstructed_data[:,j*n+i], 'o', markerfacecolor='None', markersize = 2, label = f"{rom_sizes[j]} POD modes")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.legend()
    plt.savefig(f"plot_{i}.pdf")
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