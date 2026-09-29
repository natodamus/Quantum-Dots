# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:51:20 2026

@author: natoo
"""

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from matplotlib.path import Path
from scipy.spatial import Delaunay


# Return the vertices of a regular hexagon
def hexagon_vertices(
    radius=1.0,
    center=(0.0, 0.0)
):
    xc, yc = center

    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        6,
        endpoint=False
    )

    return np.column_stack([
        xc + radius * np.cos(angles),
        yc + radius * np.sin(angles)
    ])


# Generate approximately equally spaced points along one edge
def points_along_edge(
    p1,
    p2,
    spacing
):
    edge_length = np.linalg.norm(
        p2 - p1
    )

    number_of_segments = max(
        1,
        int(
            np.ceil(
                edge_length / spacing
            )
        )
    )

    t = np.linspace(
        0.0,
        1.0,
        number_of_segments,
        endpoint=False
    )

    return (
        p1
        + t[:, None]
        * (p2 - p1)
    )


# Calculate distance from points to the hexagon boundary
def distance_to_hexagon_boundary(
    points,
    vertices
):
    minimum_distance = np.full(
        len(points),
        np.inf
    )

    for i in range(6):
        p1 = vertices[i]
        p2 = vertices[
            (i + 1) % 6
        ]

        edge = (
            p2 - p1
        )

        edge_length_squared = np.dot(
            edge,
            edge
        )

        relative = (
            points - p1
        )

        t = (
            relative
            @ edge
            / edge_length_squared
        )

        t = np.clip(
            t,
            0.0,
            1.0
        )

        closest = (
            p1
            + t[:, None]
            * edge
        )

        distance = np.linalg.norm(
            points - closest,
            axis=1
        )

        minimum_distance = np.minimum(
            minimum_distance,
            distance
        )

    return minimum_distance


# Calculate signed twice-area of each triangle
def signed_twice_areas(
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

    return (
        (p2[:, 0] - p1[:, 0])
        * (p3[:, 1] - p1[:, 1])
        - (p3[:, 0] - p1[:, 0])
        * (p2[:, 1] - p1[:, 1])
    )


# Calculate triangle areas
def triangle_areas(
    node,
    elm
):
    return (
        0.5
        * np.abs(
            signed_twice_areas(
                node,
                elm
            )
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


# Generate a triangular FEM mesh for a regular hexagon
def hexagon_mesh(
    spacing=0.08,
    radius=1.0,
    center=(0.0, 0.0)
):
    xc, yc = center

    vertices = hexagon_vertices(
        radius=radius,
        center=center
    )

    polygon = Path(
        vertices
    )

    # Generate Cartesian candidate points
    x_values = np.arange(
        xc - radius,
        xc + radius,
        spacing
    )

    y_limit = (
        np.sqrt(3.0)
        * radius
        / 2.0
    )

    y_values = np.arange(
        yc - y_limit,
        yc + y_limit,
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

    # Keep only points strictly inside the hexagon
    inside = polygon.contains_points(
        candidate_points,
        radius=-1e-12
    )

    candidate_points = candidate_points[
        inside
    ]

    # Keep Cartesian points away from the physical boundary
    wall_distance = distance_to_hexagon_boundary(
        candidate_points,
        vertices
    )

    wall_clearance = (
        0.5
        * spacing
    )

    interior_points = candidate_points[
        wall_distance
        >= wall_clearance
    ]

    # Generate explicit nodes along all six edges
    boundary_points = []

    for i in range(6):
        p1 = vertices[i]

        p2 = vertices[
            (i + 1) % 6
        ]

        boundary_points.append(
            points_along_edge(
                p1,
                p2,
                spacing
            )
        )

    boundary_points = np.vstack(
        boundary_points
    )

    # Combine interior and physical boundary nodes
    node = np.vstack([
        interior_points,
        boundary_points
    ])

    # Remove exact duplicate points
    node = np.unique(
        node,
        axis=0
    )

    # Triangulate all supplied nodes
    triangulation = Delaunay(
        node
    )

    elm = triangulation.simplices.copy()

    # Keep triangles whose centroids lie inside the hexagon
    centroids = np.mean(
        node[
            elm
        ],
        axis=1
    )

    inside_triangles = polygon.contains_points(
        centroids,
        radius=1e-12
    )

    elm = elm[
        inside_triangles
    ]

    # Remove zero or numerically degenerate triangles
    areas = triangle_areas(
        node,
        elm
    )

    area_tolerance = (
        1e-10
        * spacing**2
    )

    good_triangles = (
        areas > area_tolerance
    )

    elm = elm[
        good_triangles
    ]

    # Remove nodes that are no longer used
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
    signed_area = signed_twice_areas(
        node,
        elm
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


# Validate the hexagon mesh
def validate_hexagon_mesh(
    node,
    elm,
    boundary,
    radius,
    center=(0.0, 0.0)
):
    vertices = hexagon_vertices(
        radius=radius,
        center=center
    )

    polygon = Path(
        vertices
    )

    # Check that all nodes lie inside or on the hexagon
    node_inside = polygon.contains_points(
        node,
        radius=1e-9
    )

    if not np.all(
        node_inside
    ):
        bad_nodes = np.where(
            ~node_inside
        )[0]

        raise ValueError(
            f"{len(bad_nodes)} nodes lie outside the hexagon."
        )

    # Check element areas
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
        3.0
        * np.sqrt(3.0)
        / 2.0
        * radius**2
    )

    relative_area_error = (
        abs(
            mesh_area
            - exact_area
        )
        / exact_area
    )

    # Check that every topological boundary node lies on an edge
    boundary_points = node[
        boundary
    ]

    boundary_distance = distance_to_hexagon_boundary(
        boundary_points,
        vertices
    )

    if np.any(
        boundary_distance > 1e-9
    ):
        bad_boundary = np.where(
            boundary_distance > 1e-9
        )[0]

        raise ValueError(
            f"{len(bad_boundary)} topological boundary nodes "
            "do not lie on a physical hexagon edge."
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
    # Choose radius so the regular hexagon has area 1
    radius = np.sqrt(
        2.0
        / (
            3.0
            * np.sqrt(3.0)
        )
    )

    spacing = 0.02

    (
        node,
        elm,
        boundary
    ) = hexagon_mesh(
        spacing=spacing,
        radius=radius
    )

    mesh_info = validate_hexagon_mesh(
        node,
        elm,
        boundary,
        radius
    )

    print("Hexagon Mesh")
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
        f"Exact hexagon area      : "
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
        "Regular Hexagonal Quantum Dot FEM Mesh"
    )

    ax.set_aspect(
        "equal"
    )

    ax.legend()

    fig.tight_layout()

    plt.show()