# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:52:20 2026

@author: natoo
"""

import numpy as np
from scipy.linalg import eigh

from FEM import A_mat, B_mat
from hexagon_mesh import hexagon_mesh


# Solve the FEM eigenvalue problem for the infinite hexagonal well
def solve_hexagon(spacing, radius):
    node, elm, boundary = hexagon_mesh(
        spacing=spacing,
        radius=radius,
        center=(0.0, 0.0)
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


if __name__ == "__main__":
    # Choose the radius so the regular hexagon has area 1
    radius = np.sqrt(2.0 / (3.0 * np.sqrt(3.0)))
    spacing = 0.05

    node, elm, boundary, interior, energies, eigenvectors = solve_hexagon(
        spacing=spacing,
        radius=radius
    )

    # Check the numerical area implied by the equal-area radius
    area = (3.0 * np.sqrt(3.0) / 2.0) * radius**2

    print("Hexagonal Quantum Dot")
    print("---------------------")
    print(f"Radius       : {radius:.6f}")
    print(f"Area         : {area:.6f}")
    print(f"Mesh spacing : {spacing}")
    print(f"Nodes        : {len(node)}")
    print(f"Elements     : {len(elm)}")

    print("\nFirst 10 FEM energies")
    for state, energy in enumerate(energies[:10], start=1):
        print(f"State {state:2d}: E = {energy:.8f}")
    

