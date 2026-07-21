# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:23:46 2026

@author: natoo
"""

import numpy as np
from scipy.linalg import eigh
from FEM import A_mat, B_mat

np.set_printoptions(precision=4, suppress=True)

# Square with center interior node
node = np.array([
    [0.0, 0.0],   # 0
    [0.5, 0.0],   # 1
    [1.0, 0.0],   # 2

    [0.0, 0.5],   # 3
    [0.5, 0.5],   # 4
    [1.0, 0.5],   # 5

    [0.0, 1.0],   # 6
    [0.5, 1.0],   # 7
    [1.0, 1.0]    # 8
])

elm = np.array([
    [0,1,4], [0,4,3],
    [1,2,5], [1,5,4],
    [3,4,7], [3,7,6],
    [4,5,8], [4,8,7]
])

A = A_mat(node, elm)
B = B_mat(node, elm)

# Infinite-well boundary condition: psi = 0 on boundary
# So we only solve for interior nodes
interior = [4]

A_int = A[np.ix_(interior, interior)]
B_int = B[np.ix_(interior, interior)]

E, psi = eigh(0.5 * A_int, B_int)

print("A interior:")
print(A_int)

print("\nB interior:")
print(B_int)

print("\nEnergy eigenvalue:")
print(E)