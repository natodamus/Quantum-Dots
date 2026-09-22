# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:08:01 2026

@author: natoo
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay


# Generate a triangular FEM mesh for a circular quantum dot
def circle_mesh(
    spacing=0.08,
    radius=1.0,
    center=(0.0, 0.0),
    n_boundary=100
):
    xc, yc = center

    # Generate Cartesian candidate points
    x_values = np.arange(
        xc - radius + spacing,
        xc + radius,
        spacing
    )

    y_values = np.arange(
        yc - radius + spacing,
        yc + radius,
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

    # Keep interior points away from the explicit boundary nodes
    distance_squared = (
        (candidate_points[:, 0] - xc)**2
        + (candidate_points[:, 1] - yc)**2
    )

    interior_mask = (
        distance_squared
        < (radius - 0.25 * spacing)**2
    )

    interior_points = candidate_points[
        interior_mask
    ]

    # Generate equally spaced nodes along the circular boundary
    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        n_boundary,
        endpoint=False
    )

    boundary_points = np.column_stack([
        xc + radius * np.cos(theta),
        yc + radius * np.sin(theta)
    ])

    node = np.vstack([
        interior_points,
        boundary_points
    ])

    # Boundary nodes are added after the interior nodes
    boundary = np.arange(
        len(interior_points),
        len(node)
    )

    # Triangulate all nodes
    triangulation = Delaunay(node)
    elm = triangulation.simplices.copy()

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
    node, elm, boundary = circle_mesh(
        spacing=0.08,
        radius=1.0,
        n_boundary=100
    )

    print("Circle Mesh")
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
        "Circular Quantum Dot FEM Mesh"
    )

    plt.gca().set_aspect(
        "equal"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()