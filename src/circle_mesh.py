# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:08:01 2026

@author: natoo
"""
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from scipy.spatial import Delaunay


# Calculate triangle areas
def triangle_areas(
    node,
    elm
):
    p1 = node[
        elm[:, 0]
    ]

    p2 = node[
        elm[:, 1]
    ]

    p3 = node[
        elm[:, 2]
    ]

    twice_area = (
        (p2[:, 0] - p1[:, 0])
        * (p3[:, 1] - p1[:, 1])
        - (p3[:, 0] - p1[:, 0])
        * (p2[:, 1] - p1[:, 1])
    )

    return (
        0.5
        * np.abs(
            twice_area
        )
    )


# Find nodes on the topological mesh boundary
def find_boundary_nodes(
    elm
):
    edges = []

    for triangle in elm:
        n0, n1, n2 = triangle

        edges.append(
            tuple(
                sorted([
                    n0,
                    n1
                ])
            )
        )

        edges.append(
            tuple(
                sorted([
                    n1,
                    n2
                ])
            )
        )

        edges.append(
            tuple(
                sorted([
                    n2,
                    n0
                ])
            )
        )

    edge_counts = Counter(
        edges
    )

    boundary_edges = [
        edge
        for edge, count in edge_counts.items()
        if count == 1
    ]

    return np.unique(
        np.asarray(
            boundary_edges,
            dtype=int
        ).ravel()
    )


# Generate a triangular FEM mesh for a circular quantum dot
def circle_mesh(
    spacing=0.08,
    radius=1.0,
    center=(0.0, 0.0),
    n_boundary=None
):
    xc, yc = center

    # Match the boundary resolution approximately to the interior spacing
    if n_boundary is None:
        circumference = (
            2.0
            * np.pi
            * radius
        )

        n_boundary = max(
            32,
            int(
                np.ceil(
                    circumference
                    / spacing
                )
            )
        )

    # Generate Cartesian candidate points
    x_values = np.arange(
        xc - radius,
        xc + radius,
        spacing
    )

    y_values = np.arange(
        yc - radius,
        yc + radius,
        spacing
    )

    X, Y = np.meshgrid(
        x_values,
        y_values,
        indexing="xy"
    )

    candidate_points = np.column_stack([
        X.ravel(),
        Y.ravel()
    ])

    # Distance of each candidate point from the circle center
    radial_distance = np.sqrt(
        (candidate_points[:, 0] - xc)**2
        + (candidate_points[:, 1] - yc)**2
    )

    # Keep regular-grid points away from the explicit boundary
    wall_clearance = (
        0.5
        * spacing
    )

    interior_mask = (
        radial_distance
        <= radius - wall_clearance
    )

    interior_points = candidate_points[
        interior_mask
    ]

    # Generate equally spaced nodes along the true circular boundary
    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        n_boundary,
        endpoint=False
    )

    boundary_points = np.column_stack([
        xc
        + radius
        * np.cos(theta),
        yc
        + radius
        * np.sin(theta)
    ])

    # Combine interior and boundary points
    node = np.vstack([
        interior_points,
        boundary_points
    ])

    # Remove exact duplicate nodes if any occur
    node = np.unique(
        node,
        axis=0
    )

    # Triangulate all nodes
    triangulation = Delaunay(
        node
    )

    elm = triangulation.simplices.copy()

    # Retain triangles whose centroids lie inside the circle
    centroids = np.mean(
        node[
            elm
        ],
        axis=1
    )

    centroid_radius = np.sqrt(
        (centroids[:, 0] - xc)**2
        + (centroids[:, 1] - yc)**2
    )

    keep = (
        centroid_radius
        <= radius + 1e-12
    )

    elm = elm[
        keep
    ]

    # Remove nodes that are not used by retained triangles
    used_nodes = np.unique(
        elm.ravel()
    )

    old_to_new = -np.ones(
        len(node),
        dtype=int
    )

    old_to_new[
        used_nodes
    ] = np.arange(
        len(used_nodes)
    )

    node = node[
        used_nodes
    ]

    elm = old_to_new[
        elm
    ]

    # Ensure counterclockwise element orientation
    p1 = node[
        elm[:, 0]
    ]

    p2 = node[
        elm[:, 1]
    ]

    p3 = node[
        elm[:, 2]
    ]

    signed_area = (
        (p2[:, 0] - p1[:, 0])
        * (p3[:, 1] - p1[:, 1])
        - (p3[:, 0] - p1[:, 0])
        * (p2[:, 1] - p1[:, 1])
    )

    clockwise = (
        signed_area < 0.0
    )

    temporary = elm[
        clockwise,
        1
    ].copy()

    elm[
        clockwise,
        1
    ] = elm[
        clockwise,
        2
    ]

    elm[
        clockwise,
        2
    ] = temporary

    # Determine the actual topological boundary
    boundary = find_boundary_nodes(
        elm
    )

    return (
        node,
        elm,
        boundary
    )


# Validate the circular mesh
def validate_circle_mesh(
    node,
    elm,
    boundary,
    radius,
    center=(0.0, 0.0)
):
    xc, yc = center

    radial_distance = np.sqrt(
        (node[:, 0] - xc)**2
        + (node[:, 1] - yc)**2
    )

    # All nodes should lie inside or on the circle
    if np.any(
        radial_distance
        > radius + 1e-10
    ):
        bad_nodes = np.where(
            radial_distance
            > radius + 1e-10
        )[0]

        raise ValueError(
            f"{len(bad_nodes)} nodes lie outside the circle."
        )

    # Check triangle areas
    areas = triangle_areas(
        node,
        elm
    )

    if np.any(
        areas <= 0.0
    ):
        raise ValueError(
            "Mesh contains a zero-area triangle."
        )

    minimum_area = np.min(
        areas
    )

    mesh_area = np.sum(
        areas
    )

    exact_area = (
        np.pi
        * radius**2
    )

    relative_area_error = (
        abs(
            mesh_area
            - exact_area
        )
        / exact_area
    )

    # Every topological boundary node should lie on the circle
    boundary_radius = np.sqrt(
        (node[boundary, 0] - xc)**2
        + (node[boundary, 1] - yc)**2
    )

    if not np.allclose(
        boundary_radius,
        radius,
        rtol=0.0,
        atol=1e-9
    ):
        bad_boundary = np.where(
            np.abs(
                boundary_radius - radius
            )
            > 1e-9
        )[0]

        raise ValueError(
            f"{len(bad_boundary)} topological boundary nodes "
            "do not lie on the physical circle."
        )

    return {
        "minimum_triangle_area": minimum_area,
        "mesh_area": mesh_area,
        "exact_area": exact_area,
        "relative_area_error": relative_area_error,
        "boundary_nodes": len(
            boundary
        )
    }


if __name__ == "__main__":
    # Choose radius so the circle has area 1
    radius = (
        1.0
        / np.sqrt(
            np.pi
        )
    )

    spacing = 0.02

    (
        node,
        elm,
        boundary
    ) = circle_mesh(
        spacing=spacing,
        radius=radius,
        n_boundary=None
    )

    mesh_info = validate_circle_mesh(
        node,
        elm,
        boundary,
        radius
    )

    print("Circle Mesh")
    print("=" * 54)

    print(
        f"Radius                 : {radius:.10f}"
    )

    print(
        f"Spacing                : {spacing}"
    )

    print(
        f"Nodes                  : {len(node)}"
    )

    print(
        f"Elements               : {len(elm)}"
    )

    print(
        f"Boundary nodes          : "
        f"{mesh_info['boundary_nodes']}"
    )

    print(
        f"Minimum triangle area   : "
        f"{mesh_info['minimum_triangle_area']:.8e}"
    )

    print(
        f"Mesh area               : "
        f"{mesh_info['mesh_area']:.10f}"
    )

    print(
        f"Exact circle area       : "
        f"{mesh_info['exact_area']:.10f}"
    )

    print(
        f"Relative area error     : "
        f"{mesh_info['relative_area_error']:.6e}"
    )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.triplot(
        node[:, 0],
        node[:, 1],
        elm,
        linewidth=0.5
    )

    ax.scatter(
        node[
            boundary,
            0
        ],
        node[
            boundary,
            1
        ],
        s=12,
        label="Boundary nodes"
    )

    ax.set_xlabel(
        "x"
    )

    ax.set_ylabel(
        "y"
    )

    ax.set_title(
        "Circular Quantum Dot FEM Mesh"
    )

    ax.set_aspect(
        "equal"
    )

    ax.legend()

    fig.tight_layout()

    plt.show()