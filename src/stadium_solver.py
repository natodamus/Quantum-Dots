# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:47:17 2026

@author: natoo
"""
import numpy as np

from collections import Counter
from scipy.linalg import eigh

from FEM import A_mat, B_mat
from stadium_mesh import stadium_mesh


# Find boundary nodes by identifying edges that belong to only one triangle
def find_boundary_nodes(elm):
    edges = []

    for triangle in elm:
        n0, n1, n2 = triangle

        edges.append(tuple(sorted([n0, n1])))
        edges.append(tuple(sorted([n1, n2])))
        edges.append(tuple(sorted([n2, n0])))

    edge_counts = Counter(edges)
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]

    return np.unique(np.array(boundary_edges).flatten())


# Solve the FEM eigenvalue problem for the infinite stadium-shaped well
def solve_stadium(h, a=0.5, R=0.5, n_arc=60):
    node, elm = stadium_mesh(
        h=h,
        a=a,
        R=R,
        n_arc=n_arc
    )

    # Assemble the FEM stiffness and mass matrices
    A = A_mat(node, elm)
    B = B_mat(node, elm)

    # Apply psi = 0 on the boundary by solving only for interior nodes
    boundary = find_boundary_nodes(elm)
    interior = np.setdiff1d(np.arange(len(node)), boundary)

    A_int = A[np.ix_(interior, interior)]
    B_int = B[np.ix_(interior, interior)]

    # Solve (1/2)A psi = E B psi
    energies, eigenvectors = eigh(0.5 * A_int, B_int)

    return node, elm, boundary, interior, energies, eigenvectors


if __name__ == "__main__":
    # Use a = R and choose their value so the stadium has area 1
    a = 1.0 / np.sqrt(4.0 + np.pi)
    R = a
    h = 0.05

    node, elm, boundary, interior, energies, eigenvectors = solve_stadium(
        h=h,
        a=a,
        R=R,
        n_arc=60
    )

    # Stadium area = rectangle + two semicircular end caps
    area = 4.0 * a * R + np.pi * R**2

    print("Stadium Quantum Dot")
    print("-------------------")
    print(f"a            : {a:.6f}")
    print(f"R            : {R:.6f}")
    print(f"Area         : {area:.6f}")
    print(f"Mesh spacing : {h}")
    print(f"Nodes        : {len(node)}")
    print(f"Elements     : {len(elm)}")

    print("\nFirst 10 FEM energies")
    for state, energy in enumerate(energies[:10], start=1):
        print(f"State {state:2d}: E = {energy:.8f}")