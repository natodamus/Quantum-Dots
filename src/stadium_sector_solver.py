# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:36:33 2026

@author: natoo
"""
import numpy as np

from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


# Remove nodes that do not belong to any retained triangle
def remove_unused_nodes(node, elm):
    used_nodes = np.unique(elm.ravel())

    old_to_new = -np.ones(len(node), dtype=int)
    old_to_new[used_nodes] = np.arange(len(used_nodes))

    node = node[used_nodes]
    elm = old_to_new[elm]

    return node, elm


# Test whether points lie inside the quarter stadium
def inside_quarter_stadium(x, y, a, R, tolerance=1e-12):
    inside_rectangle = (
        (x >= -tolerance)
        & (x <= a + tolerance)
        & (y >= -tolerance)
        & (y <= R + tolerance)
    )

    inside_cap = (
        (x >= a - tolerance)
        & ((x - a)**2 + y**2 <= R**2 + tolerance)
    )

    return inside_rectangle | inside_cap


# Generate a triangular mesh for the upper-right quarter of the stadium
def quarter_stadium_mesh(h, a, R, n_arc=80):
    # Regular grid without overshooting the physical domain
    x = np.arange(0.0, a + R, h)
    y = np.arange(0.0, R, h)

    X, Y = np.meshgrid(x, y, indexing="xy")

    xf = X.ravel()
    yf = Y.ravel()

    inside = inside_quarter_stadium(
        xf,
        yf,
        a,
        R
    )

    xf = xf[inside]
    yf = yf[inside]

    # Keep interior grid points away from the physical wall
    wall_clearance = 0.5 * h

    distance_top = R - yf

    distance_curve = (
        R
        - np.sqrt(
            (xf - a)**2 + yf**2
        )
    )

    near_top_wall = (
        (xf <= a)
        & (yf > 0.0)
        & (distance_top < wall_clearance)
    )

    near_curved_wall = (
        (xf >= a)
        & (yf > 0.0)
        & (distance_curve < wall_clearance)
    )

    keep = ~(
        near_top_wall
        | near_curved_wall
    )

    grid_points = np.column_stack([
        xf[keep],
        yf[keep]
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

    # Add points along y = 0 without overshooting
    x_axis_values = np.arange(
        0.0,
        a + R,
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

    # Add points along x = 0 without overshooting
    y_axis_values = np.arange(
        0.0,
        R,
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
        a,
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

    inside_centroids = inside_quarter_stadium(
        centroids[:, 0],
        centroids[:, 1],
        a,
        R
    )

    elm = elm[inside_centroids]

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


# Calculate triangle areas
def triangle_areas(node, elm):
    p1 = node[elm[:, 0]]
    p2 = node[elm[:, 1]]
    p3 = node[elm[:, 2]]

    cross = (
        (p2[:, 0] - p1[:, 0])
        * (p3[:, 1] - p1[:, 1])
        - (p2[:, 1] - p1[:, 1])
        * (p3[:, 0] - p1[:, 0])
    )

    return 0.5 * np.abs(cross)


# Check mesh geometry and boundary classification
def validate_quarter_stadium_mesh(node, elm, a, R):
    inside = inside_quarter_stadium(
        node[:, 0],
        node[:, 1],
        a,
        R,
        tolerance=1e-9
    )

    if not np.all(inside):
        raise ValueError(
            "Mesh contains nodes outside the quarter stadium."
        )

    areas = triangle_areas(
        node,
        elm
    )

    if np.any(areas <= 0.0):
        raise ValueError(
            "Mesh contains zero-area triangles."
        )

    mesh_area = np.sum(areas)

    # Exact area of the quarter stadium
    exact_area = (
        a * R
        + 0.25 * np.pi * R**2
    )

    # Find edges that belong to only one triangle
    edges = np.vstack([
        elm[:, [0, 1]],
        elm[:, [1, 2]],
        elm[:, [2, 0]]
    ])

    edges = np.sort(
        edges,
        axis=1
    )

    unique_edges, counts = np.unique(
        edges,
        axis=0,
        return_counts=True
    )

    boundary_edges = unique_edges[
        counts == 1
    ]

    boundary_nodes = np.unique(
        boundary_edges
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

    known_boundary = np.unique(
        np.concatenate([
            x_symmetry,
            y_symmetry,
            physical_wall
        ])
    )

    unknown_boundary = np.setdiff1d(
        boundary_nodes,
        known_boundary
    )

    if len(unknown_boundary) > 0:
        raise ValueError(
            f"Mesh contains {len(unknown_boundary)} "
            "unclassified boundary nodes."
        )

    return {
        "minimum_triangle_area": np.min(areas),
        "mesh_area": mesh_area,
        "exact_area": exact_area,
        "relative_area_error": (
            abs(mesh_area - exact_area)
            / exact_area
        )
    }


# Assemble sparse FEM stiffness and mass matrices
def assemble_sparse_fem(node, elm):
    rows = []
    cols = []
    stiffness_data = []
    mass_data = []

    for triangle in elm:
        coords = node[triangle]

        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]

        signed_double_area = (
            (x2 - x1) * (y3 - y1)
            - (x3 - x1) * (y2 - y1)
        )

        area = 0.5 * abs(
            signed_double_area
        )

        if area <= 0.0:
            raise ValueError(
                "Zero-area triangle encountered."
            )

        b = np.array([
            y2 - y3,
            y3 - y1,
            y1 - y2
        ])

        c = np.array([
            x3 - x2,
            x1 - x3,
            x2 - x1
        ])

        # Local stiffness matrix
        local_A = (
            np.outer(b, b)
            + np.outer(c, c)
        ) / (4.0 * area)

        # Local consistent mass matrix
        local_B = (
            area / 12.0
        ) * np.array([
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0]
        ])

        for i in range(3):
            for j in range(3):
                rows.append(
                    triangle[i]
                )

                cols.append(
                    triangle[j]
                )

                stiffness_data.append(
                    local_A[i, j]
                )

                mass_data.append(
                    local_B[i, j]
                )

    shape = (
        len(node),
        len(node)
    )

    A = coo_matrix(
        (
            stiffness_data,
            (rows, cols)
        ),
        shape=shape
    ).tocsr()

    B = coo_matrix(
        (
            mass_data,
            (rows, cols)
        ),
        shape=shape
    ).tocsr()

    return A, B


# Solve one reflection-symmetry sector of the stadium
def solve_stadium_sector(
    sector,
    h=0.03,
    a=None,
    R=None,
    n_arc=80,
    num_eigenvalues=50
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

    # Check the mesh before solving
    mesh_info = validate_quarter_stadium_mesh(
        node,
        elm,
        a,
        R
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

    # Assemble sparse FEM stiffness and mass matrices
    A, B = assemble_sparse_fem(
        node,
        elm
    )

    A_free = A[
        free_nodes
    ][:, free_nodes]

    B_free = B[
        free_nodes
    ][:, free_nodes]

    # Check that every free degree of freedom carries positive mass
    mass_diagonal = B_free.diagonal()

    if np.any(mass_diagonal <= 0.0):
        bad_nodes = np.where(
            mass_diagonal <= 0.0
        )[0]

        raise ValueError(
            f"Mass matrix contains {len(bad_nodes)} "
            "zero or negative diagonal entries."
        )

    # Sparse solver requires fewer eigenvalues than free DOF
    k = min(
        num_eigenvalues,
        len(free_nodes) - 2
    )

    if k < 1:
        raise ValueError(
            "Not enough free degrees of freedom."
        )

    # Solve (1/2)A psi = E B psi near the bottom of the spectrum
    energies, eigenvectors = eigsh(
        0.5 * A_free,
        k=k,
        M=B_free,
        sigma=0.0,
        which="LM"
    )

    # Sort eigenvalues and eigenvectors in ascending energy
    order = np.argsort(
        energies
    )

    energies = energies[
        order
    ]

    eigenvectors = eigenvectors[
        :,
        order
    ]

    return (
        node,
        elm,
        dirichlet,
        free_nodes,
        energies,
        eigenvectors,
        mesh_info
    )


if __name__ == "__main__":
    h = 0.02
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
            eigenvectors,
            mesh_info
        ) = solve_stadium_sector(
            sector=sector,
            h=h,
            a=a,
            R=R,
            n_arc=80,
            num_eigenvalues=50
        )

        print(
            f"{sector:>8}"
            f"{len(node):10d}"
            f"{len(elm):12d}"
            f"{len(free_nodes):10d}"
            f"{energies[0]:14.6f}"
            f"{energies[1]:14.6f}"
        )

    print()
    print("Mesh diagnostics")
    print("-" * 72)

    print(
        "Minimum triangle area:",
        f"{mesh_info['minimum_triangle_area']:.8e}"
    )

    print(
        "Mesh area:",
        f"{mesh_info['mesh_area']:.10f}"
    )

    print(
        "Exact quarter area:",
        f"{mesh_info['exact_area']:.10f}"
    )

    print(
        "Relative area error:",
        f"{mesh_info['relative_area_error']:.6e}"
    )