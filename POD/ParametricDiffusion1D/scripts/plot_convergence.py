#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import sys

convergence_data = pd.read_csv(sys.argv[1], sep = ",")

plt.rcParams.update({"font.size": 8})

plt.plot(convergence_data.dt,
         convergence_data.eL2_full,
         marker = 'o',
         label = 'L2 FOM')
plt.plot(convergence_data.dt,
         convergence_data.eH1_full,
         marker = 'o',
         label = 'H1 FOM')
plt.plot(convergence_data.dt,
         convergence_data.eL2_reconstructed,
         marker = 'o',
         label = 'L2 ROM')
plt.plot(convergence_data.dt,
        convergence_data.eH1_reconstructed,
        marker = 'o',
        label = 'H1 ROM')            
plt.plot(convergence_data.dt,
         convergence_data.dt,
         '--',
         label = 'dt')
plt.plot(convergence_data.dt,
         convergence_data.dt**2,
         '--',
         label = 'dt^2')
plt.plot(convergence_data.dt,
         convergence_data.dt**3,
         '--',
         label = 'dt^3')

plt.xscale("log")
plt.yscale("log")
plt.xlabel("dt")
plt.ylabel("error")
plt.legend()

plt.savefig("convergence.pdf")