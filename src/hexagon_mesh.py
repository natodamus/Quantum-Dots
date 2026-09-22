# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:51:20 2026

@author: natoo
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.path import Path
from scipy.spatial import Delaunay


# Return the vertices of a regular hexagon
def hexagon_vertices(radius=1.0, center=(0.0, 0.0)):
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
def points_along_edge(p1, p2, spacing):
    edge_length = np.linalg.norm(p2 - p1)

    number_of_segments = max(
        1,
        int(np.ceil(edge_length / spacing))
    )

    t = np.linspace(
        0.0,
        1.0,
        number_of_segments,
        endpoint=False
    )

    return p1 + t[:, None] * (p2 - p1)


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

    polygon = Path(vertices)

    # Generate Cartesian candidate points
    x_values = np.arange(
        xc - radius,
        xc + radius + spacing,
        spacing
    )

    y_limit = np.sqrt(3.0) * radius / 2.0

    y_values = np.arange(
        yc - y_limit,
        yc + y_limit + spacing,
        spacing
    )

    X, Y = np.meshgrid(
        x_values,
        y_values
    )

    candidate_points = np.column_stack([
        X.ravel(),
        Y.ravel()
    ])

    # Keep interior grid points away from the explicit boundary nodes
    interior_mask = polygon.contains_points(
        candidate_points,
        radius=-0.2 * spacing
    )

    interior_points = candidate_points[
        interior_mask
    ]

    # Generate nodes along all six edges
    boundary_points = []

    for i in range(6):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 6]

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

    node = np.vstack([
        interior_points,
        boundary_points
    ])

    boundary = np.arange(
        len(interior_points),
        len(node)
    )

    # Triangulate all nodes
    triangulation = Delaunay(node)
    elm = triangulation.simplices.copy()

    # Retain triangles whose centroids lie inside the hexagon
    centroids = np.mean(
        node[elm],
        axis=1
    )

    inside = polygon.contains_points(
        centroids,
        radius=1e-10
    )

    elm = elm[inside]

    # Ensure counterclockwise element orientation
    p1 = node[elm[:, 0]]
    p2 = node[elm[:, 1]]
    p3 = node[elm[:, 2]]

    signed_area = (
        (p2[:, 0] - p1[:, 0])
        * (p3[:, 1] - p1[:, 1])
        - (p3[:, 0] - p1[:, 0])
        * (p2[:, 1] - p1[:, 1])
    )

    clockwise = signed_area < 0.0

    elm[clockwise, 1], elm[clockwise, 2] = (
        elm[clockwise, 2].copy(),
        elm[clockwise, 1].copy()
    )

    return node, elm, boundary


if __name__ == "__main__":
    node, elm, boundary = hexagon_mesh(
        spacing=0.08,
        radius=1.0
    )

    print("Hexagon Mesh")
    print("=" * 40)
    print(f"Nodes          : {len(node)}")
    print(f"Elements       : {len(elm)}")
    print(f"Boundary nodes : {len(boundary)}")

    plt.figure(
        figsize=(7, 7)
    )

    plt.triplot(
        node[:, 0],
        node[:, 1],
        elm,
        linewidth=0.5
    )

    plt.scatter(
        node[boundary, 0],
        node[boundary, 1],
        s=12,
        label="Boundary nodes"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        "Regular Hexagonal Quantum Dot FEM Mesh"
    )

    plt.gca().set_aspect(
        "equal"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()