# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 13:06:25 2026

@author: natoo
"""

"""
Compare the energy spectra of square, circular, and hexagonal
two-dimensional quantum dots having the same area.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

from FEM import A_mat, B_mat
from circle_mesh import circle_mesh
from hexagon_mesh import hexagon_mesh


# =========================================================
# Square mesh generator
# =========================================================

def square_mesh(N, side_length=1.0):
    """
    Generate an N x N triangular mesh on a square domain.

    Parameters
    ----------
    N : int
        Number of subdivisions along each side.

    side_length : float
        Length of each side of the square.

    Returns
    -------
    node : ndarray
        Node coordinates.

    elm : ndarray
        Triangular element connectivity.

    boundary : ndarray
        Indices of nodes on the square boundary.
    """

    node = []

    for j in range(N + 1):
        for i in range(N + 1):
            x = side_length * i / N
            y = side_length * j / N
            node.append([x, y])

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

    tolerance = 1.0e-12

    boundary = np.where(
        np.isclose(node[:, 0], 0.0, atol=tolerance)
        | np.isclose(node[:, 0], side_length, atol=tolerance)
        | np.isclose(node[:, 1], 0.0, atol=tolerance)
        | np.isclose(node[:, 1], side_length, atol=tolerance)
    )[0]

    return node, elm, boundary


# =========================================================
# General FEM eigenvalue solver
# =========================================================

def solve_geometry(node, elm, boundary, number_of_states=10):
    """
    Solve the infinite-well Schrödinger equation on a supplied mesh.

        (1/2) A psi = E B psi
    """

    A = A_mat(node, elm)
    B = B_mat(node, elm)

    all_nodes = np.arange(len(node))
    interior = np.setdiff1d(all_nodes, boundary)

    A_int = A[np.ix_(interior, interior)]
    B_int = B[np.ix_(interior, interior)]

    # Request only the lowest required eigenvalues.
    E, psi = eigh(
        0.5 * A_int,
        B_int,
        subset_by_index=[0, number_of_states - 1]
    )

    return E, psi, interior


# =========================================================
# Numerical mesh area
# =========================================================

def mesh_area(node, elm):
    """
    Compute the total area represented by the triangular mesh.
    """

    total_area = 0.0

    for triangle in elm:

        p1, p2, p3 = node[triangle]

        area = 0.5 * abs(
            (p2[0] - p1[0]) * (p3[1] - p1[1])
            - (p3[0] - p1[0]) * (p2[1] - p1[1])
        )

        total_area += area

    return total_area


# =========================================================
# Equal-area geometry dimensions
# =========================================================

target_area = 1.0

square_side = np.sqrt(target_area)

circle_radius = np.sqrt(
    target_area / np.pi
)

hexagon_radius = np.sqrt(
    2.0 * target_area / (3.0 * np.sqrt(3.0))
)

print("Equal-area dimensions")
print("-" * 45)
print(f"Target area:              {target_area:.6f}")
print(f"Square side length:       {square_side:.6f}")
print(f"Circle radius:            {circle_radius:.6f}")
print(f"Hexagon circumradius:     {hexagon_radius:.6f}")


# =========================================================
# Generate meshes
# =========================================================

# Approximately comparable spatial resolution
mesh_spacing = 0.04
square_N = round(square_side / mesh_spacing)

square_node, square_elm, square_boundary = square_mesh(
    N=square_N,
    side_length=square_side
)

circle_node, circle_elm, circle_boundary = circle_mesh(
    spacing=mesh_spacing,
    radius=circle_radius,
    center=(0.0, 0.0),
    n_boundary=100
)

hex_node, hex_elm, hex_boundary = hexagon_mesh(
    spacing=mesh_spacing,
    radius=hexagon_radius,
    center=(0.0, 0.0)
)


# =========================================================
# Check actual numerical areas
# =========================================================

square_area = mesh_area(square_node, square_elm)
circle_area = mesh_area(circle_node, circle_elm)
hexagon_area = mesh_area(hex_node, hex_elm)

print("\nNumerical mesh information")
print("-" * 70)
print(
    f"{'Geometry':<12}"
    f"{'Nodes':>10}"
    f"{'Elements':>12}"
    f"{'Mesh area':>14}"
)

print(
    f"{'Square':<12}"
    f"{len(square_node):>10}"
    f"{len(square_elm):>12}"
    f"{square_area:>14.6f}"
)

print(
    f"{'Circle':<12}"
    f"{len(circle_node):>10}"
    f"{len(circle_elm):>12}"
    f"{circle_area:>14.6f}"
)

print(
    f"{'Hexagon':<12}"
    f"{len(hex_node):>10}"
    f"{len(hex_elm):>12}"
    f"{hexagon_area:>14.6f}"
)


# =========================================================
# Solve all three geometries
# =========================================================

number_of_states = 10

square_E, _, _ = solve_geometry(
    square_node,
    square_elm,
    square_boundary,
    number_of_states
)

circle_E, _, _ = solve_geometry(
    circle_node,
    circle_elm,
    circle_boundary,
    number_of_states
)

hexagon_E, _, _ = solve_geometry(
    hex_node,
    hex_elm,
    hex_boundary,
    number_of_states
)


# =========================================================
# Print comparison table
# =========================================================

print("\nEqual-area energy comparison")
print("-" * 62)

print(
    f"{'State':>7}"
    f"{'Square':>16}"
    f"{'Circle':>16}"
    f"{'Hexagon':>16}"
)

for state in range(number_of_states):

    print(
        f"{state + 1:7d}"
        f"{square_E[state]:16.6f}"
        f"{circle_E[state]:16.6f}"
        f"{hexagon_E[state]:16.6f}"
    )


# =========================================================
# Save results to CSV
# =========================================================

with open(
    "equal_area_energy_comparison.csv",
    "w",
    newline=""
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "State",
        "Square Energy",
        "Circle Energy",
        "Hexagon Energy"
    ])

    for state in range(number_of_states):

        writer.writerow([
            state + 1,
            square_E[state],
            circle_E[state],
            hexagon_E[state]
        ])


# =========================================================
# Plot energy spectra
# =========================================================

states = np.arange(1, number_of_states + 1)

plt.figure(figsize=(8, 6))

plt.plot(
    states,
    square_E,
    marker="o",
    label="Square"
)

plt.plot(
    states,
    circle_E,
    marker="s",
    label="Circle"
)

plt.plot(
    states,
    hexagon_E,
    marker="^",
    label="Hexagon"
)

plt.xlabel("State number")
plt.ylabel("Energy")
plt.title("Energy Spectra of Equal-Area Quantum Dots")
plt.xticks(states)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "equal_area_energy_spectra.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# Ground-state comparison
# =========================================================

geometry_names = ["Square", "Circle", "Hexagon"]

ground_state_energies = [
    square_E[0],
    circle_E[0],
    hexagon_E[0]
]

plt.figure(figsize=(7, 5))

plt.bar(
    geometry_names,
    ground_state_energies
)

plt.ylabel("Ground-state energy")
plt.title("Equal-Area Ground-State Energy Comparison")
plt.tight_layout()

plt.savefig(
    "equal_area_ground_state_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()