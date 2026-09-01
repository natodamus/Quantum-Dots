# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:13:04 2026

@author: natoo
"""

"""
Solve the 2D Schrödinger equation for an infinite circular quantum dot.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

from FEM import A_mat, B_mat
from circle_mesh import circle_mesh


def solve_circle(spacing, radius, n_boundary=100):
    """
    Generate the circular mesh, assemble the FEM matrices,
    apply the boundary condition, and solve the eigenvalue problem.
    """

    node, elm, boundary = circle_mesh(
        spacing=spacing,
        radius=radius,
        center=(0.0, 0.0),
        n_boundary=n_boundary
    )

    A = A_mat(node, elm)
    B = B_mat(node, elm)

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

    return node, elm, boundary, interior, E, psi


# ---------------------------------------------------------
# Equal-area circle
# ---------------------------------------------------------

radius = 1 / np.sqrt(np.pi)

circle_area = np.pi * radius**2

print(f"Circle Radius = {radius:.6f}")
print(f"Circle Area   = {circle_area:.6f}")


# ---------------------------------------------------------
# Mesh convergence study
# ---------------------------------------------------------

print("\nCircle Mesh Convergence\n")

print(
    f"{'Spacing':>10}"
    f"{'Nodes':>10}"
    f"{'E1':>14}"
    f"{'E2':>14}"
    f"{'E3':>14}"
)

for spacing in [0.10, 0.08, 0.06, 0.05]:

    node, elm, boundary, interior, E, psi = solve_circle(
        spacing=spacing,
        radius=radius,
        n_boundary=100
    )

    print(
        f"{spacing:10.2f}"
        f"{len(node):10d}"
        f"{E[0]:14.6f}"
        f"{E[1]:14.6f}"
        f"{E[2]:14.6f}"
    )


# ---------------------------------------------------------
# Final mesh
# ---------------------------------------------------------

spacing_final = 0.05

node, elm, boundary, interior, E, psi = solve_circle(
    spacing=spacing_final,
    radius=radius,
    n_boundary=100
)


print("\nFirst 10 Circular Quantum-Dot Energies")

for state in range(10):

    print(
        f"State {state + 1:2d}: "
        f"E = {E[state]:.6f}"
    )


# ---------------------------------------------------------
# Reconstruct full wavefunctions
# ---------------------------------------------------------

psi_full = np.zeros(
    (len(node), psi.shape[1])
)

psi_full[
    interior,
    :
] = psi


# ---------------------------------------------------------
# Plot first six wavefunctions
# ---------------------------------------------------------

number_of_states = 6

for state in range(number_of_states):

    largest_index = np.argmax(
        np.abs(psi_full[:, state])
    )

    if psi_full[largest_index, state] < 0:
        psi_full[:, state] *= -1

    plt.figure(
        figsize=(6, 5)
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
        f"Circular Quantum Dot: "
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


# ---------------------------------------------------------
# Plot first six probability densities
# ---------------------------------------------------------

for state in range(number_of_states):

    probability_density = (
        np.abs(psi_full[:, state])**2
    )

    plt.figure(
        figsize=(6, 5)
    )

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
        f"Circular Quantum Dot Probability Density: "
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