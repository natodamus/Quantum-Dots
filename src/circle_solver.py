# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:13:04 2026

@author: natoo
"""

import numpy as np
from scipy.linalg import eigh
from scipy.special import jn_zeros

from FEM import A_mat, B_mat
from circle_mesh import circle_mesh


# Solve the FEM eigenvalue problem for the infinite circular well
def solve_circle(spacing, radius, n_boundary=100):
    node, elm, boundary = circle_mesh(
        spacing=spacing,
        radius=radius,
        center=(0.0, 0.0),
        n_boundary=n_boundary
    )

    # Assemble the FEM stiffness and mass matrices
    A = A_mat(node, elm)
    B = B_mat(node, elm)

    # Apply psi = 0 on the boundary by solving only for interior nodes
    interior = np.setdiff1d(np.arange(len(node)), boundary)

    A_int = A[np.ix_(interior, interior)]
    B_int = B[np.ix_(interior, interior)]

    # Solve (1/2)A psi = E B psi
    energies, eigenvectors = eigh(0.5 * A_int, B_int)

    return node, elm, boundary, interior, energies, eigenvectors


# Analytical energy levels for a 2D infinite circular well
def exact_circle_energy(m, n, radius):
    zero = jn_zeros(m, n)[-1]
    return zero**2 / (2.0 * radius**2)


if __name__ == "__main__":
    # Choose the radius so the circle has the same area as a unit square
    radius = 1.0 / np.sqrt(np.pi)
    spacing = 0.05

    node, elm, boundary, interior, energies, eigenvectors = solve_circle(
        spacing=spacing,
        radius=radius,
        n_boundary=100
    )

    # Compare the numerical ground state with the analytical Bessel solution
    exact_ground = exact_circle_energy(0, 1, radius)
    error = abs(energies[0] - exact_ground) / exact_ground * 100.0

    print("Circular Quantum Dot")
    print("--------------------")
    print(f"Radius            : {radius:.6f}")
    print(f"Area              : {np.pi * radius**2:.6f}")
    print(f"Mesh spacing      : {spacing}")
    print(f"Nodes             : {len(node)}")
    print(f"Elements          : {len(elm)}")
    print(f"FEM ground energy : {energies[0]:.8f}")
    print(f"Exact energy      : {exact_ground:.8f}")
    print(f"Relative error    : {error:.4f}%")

    print("\nFirst 10 FEM energies")
    for state, energy in enumerate(energies[:10], start=1):
        print(f"State {state:2d}: E = {energy:.8f}")