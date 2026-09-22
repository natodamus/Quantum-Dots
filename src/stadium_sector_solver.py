# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:36:33 2026

@author: natoo
"""

import numpy as np

from scipy.linalg import eigh
from scipy.spatial import Delaunay

from FEM import A_mat, B_mat


# Remove nodes that do not belong to any retained triangle
def remove_unused_nodes(node, elm):
    used_nodes = np.unique(elm.ravel())

    old_to_new = -np.ones(len(node), dtype=int)
    old_to_new[used_nodes] = np.arange(len(used_nodes))

    node = node[used_nodes]
    elm = old_to_new[elm]

    return node, elm


# Generate a triangular mesh for the upper-right quarter of the stadium
def quarter_stadium_mesh(h, a, R, n_arc=60):
    # Sample points on a regular grid
    x = np.arange(0.0, a + R + 0.5 * h, h)
    y = np.arange(0.0, R + 0.5 * h, h)

    X, Y = np.meshgrid(x, y, indexing="xy")

    xf = X.ravel()
    yf = Y.ravel()

    # Quarter stadium = rectangle plus upper-right quarter-circle cap
    inside_rectangle = (
        (xf >= 0.0)
        & (xf <= a)
        & (yf >= 0.0)
        & (yf <= R)
    )

    inside_cap = (
        (xf >= a)
        & ((xf - a)**2 + yf**2 <= R**2)
    )

    inside = inside_rectangle | inside_cap

    grid_points = np.column_stack([
        xf[inside],
        yf[inside]
    ])

    # Add points along the curved physical boundary
    theta = np.linspace(
        0.0,
        0.5 * np.pi,
        n_arc
    )

    arc_points = np.column_stack([
        a + R * np.cos(theta),
        R * np.sin(theta)
    ])

    # Add points along y = 0
    x_axis_values = np.arange(
        0.0,
        a + R + 0.5 * h,
        h
    )

    x_axis_points = np.column_stack([
        x_axis_values,
        np.zeros_like(x_axis_values)
    ])

    # Include the exact rightmost point
    x_axis_points = np.vstack([
        x_axis_points,
        [a + R, 0.0]
    ])

    # Add points along x = 0
    y_axis_values = np.arange(
        0.0,
        R + 0.5 * h,
        h
    )

    y_axis_points = np.column_stack([
        np.zeros_like(y_axis_values),
        y_axis_values
    ])

    # Include the exact upper-left point
    y_axis_points = np.vstack([
        y_axis_points,
        [0.0, R]
    ])

    # Add points along the horizontal physical wall
    top_values = np.arange(
        0.0,
        a + 0.5 * h,
        h
    )

    top_points = np.column_stack([
        top_values,
        np.full_like(top_values, R)
    ])

    # Include the exact junction between straight and curved boundaries
    top_points = np.vstack([
        top_points,
        [a, R]
    ])

    points = np.vstack([
        grid_points,
        arc_points,
        x_axis_points,
        y_axis_points,
        top_points
    ])

    # Remove duplicate points
    node = np.unique(
        np.round(points, decimals=12),
        axis=0
    )

    triangulation = Delaunay(node)
    elm = triangulation.simplices.copy()

    # Keep triangles whose centroids lie inside the quarter stadium
    centroids = np.mean(
        node[elm],
        axis=1
    )

    xc = centroids[:, 0]
    yc = centroids[:, 1]

    inside_rectangle = (
        (xc >= 0.0)
        & (xc <= a)
        & (yc >= 0.0)
        & (yc <= R)
    )

    inside_cap = (
        (xc >= a)
        & ((xc - a)**2 + yc**2 <= R**2)
    )

    elm = elm[
        inside_rectangle | inside_cap
    ]

    # Remove orphan nodes created when exterior triangles were discarded
    node, elm = remove_unused_nodes(
        node,
        elm
    )

    return node, elm


# Identify symmetry axes and the physical hard-wall boundary
def quarter_stadium_boundaries(node, a, R, tolerance=1e-8):
    x = node[:, 0]
    y = node[:, 1]

    # x = 0 controls parity under x reflection
    x_symmetry = np.where(
        np.isclose(
            x,
            0.0,
            atol=tolerance
        )
    )[0]

    # y = 0 controls parity under y reflection
    y_symmetry = np.where(
        np.isclose(
            y,
            0.0,
            atol=tolerance
        )
    )[0]

    # Straight physical wall at y = R
    top_wall = (
        (x <= a + tolerance)
        & np.isclose(
            y,
            R,
            atol=tolerance
        )
    )

    # Curved physical wall
    radius_from_center = np.sqrt(
        (x - a)**2 + y**2
    )

    curved_wall = (
        (x >= a - tolerance)
        & np.isclose(
            radius_from_center,
            R,
            atol=1e-7
        )
    )

    physical_wall = np.where(
        top_wall | curved_wall
    )[0]

    return (
        x_symmetry,
        y_symmetry,
        physical_wall
    )


# Determine which symmetry axes use Dirichlet conditions
def sector_conditions(sector):
    conditions = {
        "++": (False, False),
        "+-": (False, True),
        "-+": (True, False),
        "--": (True, True)
    }

    if sector not in conditions:
        raise ValueError(
            "sector must be '++', '+-', '-+', or '--'"
        )

    # True = Dirichlet, False = natural Neumann
    return conditions[sector]


# Solve one reflection-symmetry sector of the stadium
def solve_stadium_sector(
    sector,
    h=0.03,
    a=None,
    R=None,
    n_arc=60
):
    if a is None:
        a = 1.0 / np.sqrt(4.0 + np.pi)

    if R is None:
        R = a

    node, elm = quarter_stadium_mesh(
        h=h,
        a=a,
        R=R,
        n_arc=n_arc
    )

    (
        x_symmetry,
        y_symmetry,
        physical_wall
    ) = quarter_stadium_boundaries(
        node,
        a,
        R
    )

    x_dirichlet, y_dirichlet = sector_conditions(
        sector
    )

    # Physical stadium boundary always satisfies psi = 0
    dirichlet = list(physical_wall)

    # Odd x parity requires psi = 0 along x = 0
    if x_dirichlet:
        dirichlet.extend(
            x_symmetry
        )

    # Odd y parity requires psi = 0 along y = 0
    if y_dirichlet:
        dirichlet.extend(
            y_symmetry
        )

    dirichlet = np.unique(
        np.asarray(
            dirichlet,
            dtype=int
        )
    )

    free_nodes = np.setdiff1d(
        np.arange(len(node)),
        dirichlet
    )

    # Assemble FEM stiffness and mass matrices
    A = A_mat(
        node,
        elm
    )

    B = B_mat(
        node,
        elm
    )

    A_free = A[
        np.ix_(
            free_nodes,
            free_nodes
        )
    ]

    B_free = B[
        np.ix_(
            free_nodes,
            free_nodes
        )
    ]

    # Check that every free degree of freedom carries nonzero mass
    mass_diagonal = np.diag(B_free)

    if np.any(mass_diagonal <= 0.0):
        bad_nodes = np.where(
            mass_diagonal <= 0.0
        )[0]

        raise ValueError(
            f"Mass matrix contains {len(bad_nodes)} "
            "zero or negative diagonal entries."
        )

    # Solve (1/2)A psi = E B psi
    energies, eigenvectors = eigh(
        0.5 * A_free,
        B_free
    )

    return (
        node,
        elm,
        dirichlet,
        free_nodes,
        energies,
        eigenvectors
    )


if __name__ == "__main__":
    h = 0.03
    a = 1.0 / np.sqrt(4.0 + np.pi)
    R = a

    print("Quarter-Stadium Symmetry Sectors")
    print("=" * 72)

    print(
        f"{'Sector':>8}"
        f"{'Nodes':>10}"
        f"{'Elements':>12}"
        f"{'DOF':>10}"
        f"{'E1':>14}"
        f"{'E2':>14}"
    )

    for sector in [
        "++",
        "+-",
        "-+",
        "--"
    ]:
        (
            node,
            elm,
            dirichlet,
            free_nodes,
            energies,
            eigenvectors
        ) = solve_stadium_sector(
            sector=sector,
            h=h,
            a=a,
            R=R,
            n_arc=60
        )

        print(
            f"{sector:>8}"
            f"{len(node):10d}"
            f"{len(elm):12d}"
            f"{len(free_nodes):10d}"
            f"{energies[0]:14.6f}"
            f"{energies[1]:14.6f}"
        )