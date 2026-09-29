# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:43:31 2026

@author: natoo
"""

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from scipy.spatial import Delaunay


# Check whether points lie inside the stadium
def inside_stadium(
    x,
    y,
    a,
    R,
    tolerance=1e-12
):
    x = np.asarray(
        x
    )

    y = np.asarray(
        y
    )

    rectangle = (
        (np.abs(x) <= a + tolerance)
        & (np.abs(y) <= R + tolerance)
    )

    left_cap = (
        (x < -a)
        & (
            (x + a)**2
            + y**2
            <= R**2 + tolerance
        )
    )

    right_cap = (
        (x > a)
        & (
            (x - a)**2
            + y**2
            <= R**2 + tolerance
        )
    )

    return (
        rectangle
        | left_cap
        | right_cap
    )


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
def topological_boundary_nodes(
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


# Generate a triangular FEM mesh for a stadium
def stadium_mesh(
    h=0.10,
    a=0.5,
    R=0.5,
    n_arc=80
):
    # Generate Cartesian candidate points without overshooting
    x_values = np.arange(
        -(a + R),
        a + R,
        h
    )

    y_values = np.arange(
        -R,
        R,
        h
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

    x = candidate_points[
        :,
        0
    ]

    y = candidate_points[
        :,
        1
    ]

    # Keep only points inside the stadium
    interior_mask = inside_stadium(
        x,
        y,
        a,
        R
    )

    candidate_points = candidate_points[
        interior_mask
    ]

    x = candidate_points[
        :,
        0
    ]

    y = candidate_points[
        :,
        1
    ]

    # Remove regular-grid points too close to the physical wall
    wall_clearance = (
        0.5
        * h
    )

    distance_top = (
        R - y
    )

    distance_bottom = (
        y + R
    )

    near_top = (
        (np.abs(x) <= a)
        & (distance_top < wall_clearance)
    )

    near_bottom = (
        (np.abs(x) <= a)
        & (distance_bottom < wall_clearance)
    )

    # Distance from the centers of the circular end caps
    right_radius = np.sqrt(
        (x - a)**2
        + y**2
    )

    left_radius = np.sqrt(
        (x + a)**2
        + y**2
    )

    near_right_cap = (
        (x > a)
        & (
            R - right_radius
            < wall_clearance
        )
    )

    near_left_cap = (
        (x < -a)
        & (
            R - left_radius
            < wall_clearance
        )
    )

    near_wall = (
        near_top
        | near_bottom
        | near_right_cap
        | near_left_cap
    )

    interior_points = candidate_points[
        ~near_wall
    ]

    # Generate points along the straight upper and lower walls
    n_line = max(
        2,
        int(
            np.ceil(
                2.0 * a / h
            )
        ) + 1
    )

    x_line = np.linspace(
        -a,
        a,
        n_line
    )

    top_boundary = np.column_stack([
        x_line,
        np.full_like(
            x_line,
            R
        )
    ])

    bottom_boundary = np.column_stack([
        x_line,
        np.full_like(
            x_line,
            -R
        )
    ])

    # Generate points along the right semicircular wall
    theta_right = np.linspace(
        -0.5 * np.pi,
        0.5 * np.pi,
        n_arc
    )

    right_boundary = np.column_stack([
        a
        + R
        * np.cos(
            theta_right
        ),
        R
        * np.sin(
            theta_right
        )
    ])

    # Generate points along the left semicircular wall
    theta_left = np.linspace(
        0.5 * np.pi,
        1.5 * np.pi,
        n_arc
    )

    left_boundary = np.column_stack([
        -a
        + R
        * np.cos(
            theta_left
        ),
        R
        * np.sin(
            theta_left
        )
    ])

    # Combine interior and exact boundary points
    points = np.vstack([
        interior_points,
        top_boundary,
        bottom_boundary,
        right_boundary,
        left_boundary
    ])

    # Remove duplicate nodes
    node = np.unique(
        np.round(
            points,
            decimals=12
        ),
        axis=0
    )

    # Triangulate the stadium
    triangulation = Delaunay(
        node
    )

    elm = triangulation.simplices.copy()

    # Remove any triangle whose centroid lies outside the stadium
    centroids = np.mean(
        node[
            elm
        ],
        axis=1
    )

    keep = inside_stadium(
        centroids[:, 0],
        centroids[:, 1],
        a,
        R
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

    return (
        node,
        elm
    )


# Validate the stadium mesh
def validate_stadium_mesh(
    node,
    elm,
    a,
    R
):
    # Every retained node should lie inside the stadium
    inside = inside_stadium(
        node[:, 0],
        node[:, 1],
        a,
        R
    )

    if not np.all(
        inside
    ):
        bad_nodes = np.where(
            ~inside
        )[0]

        raise ValueError(
            f"{len(bad_nodes)} mesh nodes lie outside the stadium."
        )

    # Every triangle should have positive area
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

    mesh_area = np.sum(
        areas
    )

    exact_area = (
        4.0
        * a
        * R
        + np.pi
        * R**2
    )

    relative_area_error = (
        abs(
            mesh_area
            - exact_area
        )
        / exact_area
    )

    # Check that topological boundary nodes lie on the physical wall
    boundary = topological_boundary_nodes(
        elm
    )

    xb = node[
        boundary,
        0
    ]

    yb = node[
        boundary,
        1
    ]

    tolerance = 1e-7

    top_wall = (
        (np.abs(xb) <= a + tolerance)
        & np.isclose(
            yb,
            R,
            atol=tolerance
        )
    )

    bottom_wall = (
        (np.abs(xb) <= a + tolerance)
        & np.isclose(
            yb,
            -R,
            atol=tolerance
        )
    )

    right_radius = np.sqrt(
        (xb - a)**2
        + yb**2
    )

    right_wall = (
        (xb >= a - tolerance)
        & np.isclose(
            right_radius,
            R,
            atol=tolerance
        )
    )

    left_radius = np.sqrt(
        (xb + a)**2
        + yb**2
    )

    left_wall = (
        (xb <= -a + tolerance)
        & np.isclose(
            left_radius,
            R,
            atol=tolerance
        )
    )

    physical_wall = (
        top_wall
        | bottom_wall
        | right_wall
        | left_wall
    )

    if not np.all(
        physical_wall
    ):
        bad_boundary = boundary[
            ~physical_wall
        ]

        raise ValueError(
            f"{len(bad_boundary)} topological boundary nodes "
            "are not on the physical stadium wall."
        )

    return {
        "minimum_triangle_area": np.min(
            areas
        ),
        "mesh_area": mesh_area,
        "exact_area": exact_area,
        "relative_area_error": relative_area_error,
        "boundary_nodes": len(
            boundary
        )
    }


if __name__ == "__main__":
    a = 1.0 / np.sqrt(
        4.0 + np.pi
    )

    R = a
    h = 0.02

    node, elm = stadium_mesh(
        h=h,
        a=a,
        R=R,
        n_arc=80
    )

    mesh_info = validate_stadium_mesh(
        node,
        elm,
        a,
        R
    )

    print("Stadium Mesh")
    print("=" * 54)

    print(
        f"Nodes                 : {len(node)}"
    )

    print(
        f"Elements              : {len(elm)}"
    )

    print(
        f"Boundary nodes         : "
        f"{mesh_info['boundary_nodes']}"
    )

    print(
        f"Minimum triangle area  : "
        f"{mesh_info['minimum_triangle_area']:.8e}"
    )

    print(
        f"Mesh area              : "
        f"{mesh_info['mesh_area']:.10f}"
    )

    print(
        f"Exact stadium area     : "
        f"{mesh_info['exact_area']:.10f}"
    )

    print(
        f"Relative area error    : "
        f"{mesh_info['relative_area_error']:.6e}"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.triplot(
        node[:, 0],
        node[:, 1],
        elm,
        linewidth=0.5
    )

    ax.scatter(
        node[:, 0],
        node[:, 1],
        s=5
    )

    ax.set_xlabel(
        "x"
    )

    ax.set_ylabel(
        "y"
    )

    ax.set_title(
        f"Stadium Mesh, h = {h}"
    )

    ax.set_aspect(
        "equal"
    )

    fig.tight_layout()

    plt.show()