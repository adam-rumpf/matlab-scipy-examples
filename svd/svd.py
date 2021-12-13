# svd.py
#
# Demonstrating how to compute the singular value decomposition (SVD) of a
# given matrix, and then using the SVD to illustrate the effect of the
# matrix on the unit ball.

import numpy as np

# Define example matrices
a = np.array([[1, 2], [1, 3]])
b = np.array([[2, 0], [0, 1]])
c = np.array([[0, 2], [1, 1]])
