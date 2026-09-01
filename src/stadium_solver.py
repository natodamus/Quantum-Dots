# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:47:17 2026

@author: natoo
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import eigh

from FEM import A_mat, B_mat
from stadium_mesh import stadium_mesh


def find_boundary_nodes(elm):
    """
    Find boundary nodes from the triangular mesh.

    Interior edges belong to two triangles.
    Boundary edges belong to only one triangle.
    """

    edges = []

    for triangle in elm:

        n0, n1, n2 = triangle

        edges.append(tuple(sorted([n0, n1])))
        edges.append(tuple(sorted([n1, n2])))
        edges.append(tuple(sorted([n2, n0])))

    boundary_edges = []

    for edge in edges:

        if edges.count(edge) == 1:
            boundary_edges.append(edge)

    boundary = np.unique(
        np.array(boundary_edges).flatten()
    )

    return boundary


def solve_stadium(h, a=0.5, R=0.5, n_arc=60):
    """
    Generate the stadium mesh, assemble the FEM matrices,
    apply the boundary condition, and solve the eigenvalue problem.
    """

    node, elm = stadium_mesh(
        h=h,
        a=a,
        R=R,
        n_arc=n_arc
    )

    A = A_mat(node, elm)
    B = B_mat(node, elm)

    boundary = find_boundary_nodes(elm)

    all_nodes = np.arange(len(node))

    interior = np.setdiff1d(
        all_nodes,
        boundary
    )

    A_int = A[
        np.ix_(interior, interior)
    ]

    B_int = B[
        np.ix_(interior, interior)
    ]

    E, psi = eigh(
        0.5 * A_int,
        B_int
    )

    return node, elm, interior, E, psi


# --------------------------------------------------
# Stadium parameters
# --------------------------------------------------

a = 1 / np.sqrt(4+np.pi)
R = a


# --------------------------------------------------
# Mesh convergence study
# --------------------------------------------------

print("Stadium Mesh Convergence\n")

print(
    f"{'h':>8}"
    f"{'Nodes':>10}"
    f"{'E1':>14}"
    f"{'E2':>14}"
    f"{'E3':>14}"
)

for h in [0.10, 0.08, 0.06, 0.05]:

    node, elm, interior, E, psi = solve_stadium(
        h=h,
        a=a,
        R=R,
        n_arc=60
    )

    print(
        f"{h:8.2f}"
        f"{len(node):10d}"
        f"{E[0]:14.6f}"
        f"{E[1]:14.6f}"
        f"{E[2]:14.6f}"
    )


# --------------------------------------------------
# Final mesh
# --------------------------------------------------

h_final = 0.05

node, elm, interior, E, psi = solve_stadium(
    h=h_final,
    a= a,
    R= R,
    n_arc=60
)


print("\nFirst 10 Stadium Energy Levels")

for state in range(10):

    print(
        f"State {state + 1:2d}: "
        f"E = {E[state]:.6f}"
    )


# --------------------------------------------------
# Reconstruct full wavefunctions
# --------------------------------------------------

psi_full = np.zeros(
    (len(node), psi.shape[1])
)

psi_full[
    interior,
    :
] = psi


# --------------------------------------------------
# Plot first 6 eigenstates
# --------------------------------------------------

for state in range(6):

    if np.sum(
        psi_full[:, state]
    ) < 0:

        psi_full[
            :,
            state
        ] *= -1

    plt.figure(
        figsize=(9, 5)
    )

    plt.tripcolor(
        node[:, 0],
        node[:, 1],
        elm,
        psi_full[:, state],
        shading="gouraud"
    )

    plt.colorbar(
        label=r"$\psi(x,y)$"
    )

    plt.title(
        f"Stadium Quantum Dot: "
        f"State {state + 1}, "
        f"E = {E[state]:.4f}"
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.gca().set_aspect(
        "equal"
    )

    plt.tight_layout()

    plt.show()