# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:39:01 2026

@author: natoo
"""

import numpy as np

from scipy.linalg import eigh
from FEM import A_mat, B_mat

def square_mesh(N):
    """
    Create an N x N square grid on [0,1] x [0,1].
    Each square is split into two triangles.
    """

    node = []

    for j in range(N + 1):
        for i in range(N + 1):
            node.append([i / N, j / N])

    node = np.array(node)

    elm = []

    for j in range(N):
        for i in range(N):
            n0 = j * (N + 1) + i
            n1 = n0 + 1
            n2 = n0 + (N + 1)
            n3 = n2 + 1

            elm.append([n0, n1, n3])
            elm.append([n0, n3, n2])

    elm = np.array(elm)

    return node, elm


exact = np.pi**2

print(f"{'N':>5} {'E1 FEM':>12} {'Error (%)':>12}")

for N in [5, 10, 20, 30]:

    node, elm = square_mesh(N)

    A = A_mat(node, elm)
    B = B_mat(node, elm)

    boundary = np.where(
        (node[:,0]==0) |
        (node[:,0]==1) |
        (node[:,1]==0) |
        (node[:,1]==1)
    )[0]

    all_nodes = np.arange(len(node))
    interior = np.setdiff1d(all_nodes, boundary)

    A_int = A[np.ix_(interior, interior)]
    B_int = B[np.ix_(interior, interior)]

    E, psi = eigh(0.5*A_int, B_int)

    error = abs(E[0]-exact)/exact*100

    print(f"{N:5d} {E[0]:12.6f} {error:12.3f}")
    
    import matplotlib.pyplot as plt

N_values = np.array([5, 10, 20, 30])
E1_values = np.array([10.861103, 10.114213, 9.930552, 9.896676])
exact_E1 = np.pi**2

plt.figure(figsize=(7, 5))
plt.plot(N_values, E1_values, 'o-', label="FEM ground state")
plt.axhline(exact_E1, linestyle='--', label="Exact $E_{11}=\\pi^2$")
plt.xlabel("Mesh refinement N")
plt.ylabel("Ground-state energy")
plt.title("Convergence of FEM Ground-State Energy")
plt.legend()
plt.grid(True)
plt.show()