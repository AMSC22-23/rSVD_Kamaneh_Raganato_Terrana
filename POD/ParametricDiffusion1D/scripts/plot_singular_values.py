#!/usr/bin/env python
from scipy.io import mmread
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Read the matrix containing the singular values
sigma = mmread(sys.argv[1])
s = np.array(sigma)

# Plot the singular values decay
plt.rcParams.update({"font.size": 8})

# fig, axs = plt.subplots(1, 3, figsize=(18, 6))
plt.semilogy(s, 'o-')
plt.title('Singular values')

# axs[1].semilogx(np.cumsum(s)/np.sum(s), 'o-')
# axs[1].set_title('Cumulative fraction')

# axs[2].semilogx(np.cumsum(s**2)/np.sum(s**2), 'o-')
# axs[2].set_title('Explained variance')

plt.savefig("singular_values.pdf")
plt.close()

