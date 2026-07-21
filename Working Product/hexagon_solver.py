# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:52:20 2026

@author: natoo
"""

"""
Solve the 2D Schrödinger equation for an infinite
regular hexagonal quantum dot.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

from FEM import A_mat, B_mat
from hexagon_mesh import hexagon_mesh


# ---------------------------------------------------------
# 1. Generate hexagonal mesh
# ---------------------------------------------------------

spacing = 0.08
radius = 1.0

node, elm, boundary = hexagon_mesh(
    spacing=spacing,
    radius=radius,
    center=(0.0, 0.0)
)

print("Number of nodes:", len(node))
print("Number of elements:", len(elm))
print("Number of boundary nodes:", len(boundary))


# ---------------------------------------------------------
# 2. Assemble FEM matrices
# ---------------------------------------------------------

A = A_mat(node, elm)
B = B_mat(node, elm)


# ---------------------------------------------------------
# 3. Apply infinite-well boundary conditions
# ---------------------------------------------------------

all_nodes = np.arange(len(node))
interior = np.setdiff1d(
    all_nodes,
    boundary
)

A_int = A[np.ix_(interior, interior)]
B_int = B[np.ix_(interior, interior)]


# ---------------------------------------------------------
# 4. Solve generalized eigenvalue problem
#
#     (1/2) A psi = E B psi
# ---------------------------------------------------------

E, psi = eigh(
    0.5 * A_int,
    B_int
)

print("\nFirst 10 hexagonal quantum-dot energies:")

for state in range(10):
    print(
        f"State {state + 1:2d}: "
        f"E = {E[state]:.6f}"
    )


# ---------------------------------------------------------
# 5. Reconstruct full wavefunctions
# ---------------------------------------------------------

psi_full = np.zeros(
    (len(node), psi.shape[1])
)

psi_full[interior, :] = psi


# ---------------------------------------------------------
# 6. Plot first six wavefunctions
# ---------------------------------------------------------

number_of_states = 6

for state in range(number_of_states):

    largest_index = np.argmax(
        np.abs(psi_full[:, state])
    )

    if psi_full[largest_index, state] < 0:
        psi_full[:, state] *= -1

    plt.figure(figsize=(6, 5))

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
        f"Hexagonal Quantum Dot: "
        f"State {state + 1}, "
        f"E = {E[state]:.4f}"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 7. Plot first six probability densities
# ---------------------------------------------------------

for state in range(number_of_states):

    probability_density = (
        np.abs(psi_full[:, state])**2
    )

    plt.figure(figsize=(6, 5))

    plt.tripcolor(
        node[:, 0],
        node[:, 1],
        elm,
        probability_density,
        shading="gouraud"
    )

    plt.colorbar(
        label=r"$|\psi(x,y)|^2$"
    )

    plt.title(
        f"Hexagonal Quantum Dot Probability Density: "
        f"State {state + 1}, "
        f"E = {E[state]:.4f}"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()