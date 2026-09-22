# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:39:01 2026

@author: natoo
"""

import numpy as np
from scipy.linalg import eigh

from FEM import A_mat, B_mat


# Generate a triangular mesh for a square domain
def square_mesh(N, length=1.0):
    node = []

    for j in range(N + 1):
        for i in range(N + 1):
            node.append([length * i / N, length * j / N])

    node = np.asarray(node, dtype=float)
    elm = []

    # Divide each square mesh cell into two triangular elements
    for j in range(N):
        for i in range(N):
            n0 = j * (N + 1) + i
            n1 = n0 + 1
            n2 = n0 + (N + 1)
            n3 = n2 + 1

            elm.append([n0, n1, n3])
            elm.append([n0, n3, n2])

    return node, np.asarray(elm, dtype=int)


# Identify nodes along the boundary of the square
def square_boundary_nodes(node, length=1.0):
    x = node[:, 0]
    y = node[:, 1]

    boundary_mask = (
        np.isclose(x, 0.0)
        | np.isclose(x, length)
        | np.isclose(y, 0.0)
        | np.isclose(y, length)
    )

    return np.where(boundary_mask)[0]


# Solve the FEM eigenvalue problem for the infinite square well
def solve_square(N, length=1.0):
    node, elm = square_mesh(N, length)

    # Assemble the FEM stiffness and mass matrices
    A = A_mat(node, elm)
    B = B_mat(node, elm)

    # Apply psi = 0 on the boundary by solving only for interior nodes
    boundary = square_boundary_nodes(node, length)
    interior = np.setdiff1d(np.arange(len(node)), boundary)

    A_int = A[np.ix_(interior, interior)]
    B_int = B[np.ix_(interior, interior)]

    # Solve (1/2)A psi = E B psi
    energies, eigenvectors = eigh(0.5 * A_int, B_int)

    return node, elm, boundary, interior, energies, eigenvectors


# Analytical energy levels for a 2D infinite square well
def exact_square_energy(nx, ny, length=1.0):
    return (np.pi**2 / (2.0 * length**2)) * (nx**2 + ny**2)


if __name__ == "__main__":
    N = 30
    length = 1.0

    node, elm, boundary, interior, energies, eigenvectors = solve_square(N, length)

    # Compare the numerical ground state with E_11 = pi^2 for a unit square
    exact_ground = exact_square_energy(1, 1, length)
    error = abs(energies[0] - exact_ground) / exact_ground * 100.0

    print("Square Quantum Dot")
    print("------------------")
    print(f"Mesh subdivisions : {N}")
    print(f"Nodes             : {len(node)}")
    print(f"Elements          : {len(elm)}")
    print(f"FEM ground energy : {energies[0]:.8f}")
    print(f"Exact energy      : {exact_ground:.8f}")
    print(f"Relative error    : {error:.4f}%")

    print("\nFirst 10 FEM energies")
    for state, energy in enumerate(energies[:10], start=1):
        print(f"State {state:2d}: E = {energy:.8f}")