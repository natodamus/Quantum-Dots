# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:43:31 2026

@author: natoo
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay


# Check whether points lie inside the stadium
def inside_stadium(x, y, a, R):
    x = np.asarray(x)
    y = np.asarray(y)

    rectangle = (
        (np.abs(x) <= a)
        & (np.abs(y) <= R)
    )

    left_cap = (
        (x < -a)
        & ((x + a)**2 + y**2 <= R**2)
    )

    right_cap = (
        (x > a)
        & ((x - a)**2 + y**2 <= R**2)
    )

    return rectangle | left_cap | right_cap


# Generate a triangular FEM mesh for a stadium
def stadium_mesh(
    h=0.10,
    a=0.5,
    R=0.5,
    n_arc=40
):
    # Generate Cartesian candidate points
    x_values = np.arange(
        -(a + R),
        a + R + h,
        h
    )

    y_values = np.arange(
        -R,
        R + h,
        h
    )

    X, Y = np.meshgrid(
        x_values,
        y_values
    )

    candidate_points = np.column_stack([
        X.ravel(),
        Y.ravel()
    ])

    interior_mask = inside_stadium(
        candidate_points[:, 0],
        candidate_points[:, 1],
        a,
        R
    )

    interior_points = candidate_points[
        interior_mask
    ]

    # Generate points along the straight upper and lower boundaries
    n_line = int(
        (2.0 * a) / h
    ) + 1

    x_line = np.linspace(
        -a,
        a,
        n_line
    )

    top_boundary = np.column_stack([
        x_line,
        np.full_like(x_line, R)
    ])

    bottom_boundary = np.column_stack([
        x_line,
        np.full_like(x_line, -R)
    ])

    # Generate points along the right semicircle
    theta_right = np.linspace(
        -0.5 * np.pi,
        0.5 * np.pi,
        n_arc
    )

    right_boundary = np.column_stack([
        a + R * np.cos(theta_right),
        R * np.sin(theta_right)
    ])

    # Generate points along the left semicircle
    theta_left = np.linspace(
        0.5 * np.pi,
        1.5 * np.pi,
        n_arc
    )

    left_boundary = np.column_stack([
        -a + R * np.cos(theta_left),
        R * np.sin(theta_left)
    ])

    # Combine all mesh points
    node = np.vstack([
        interior_points,
        top_boundary,
        bottom_boundary,
        right_boundary,
        left_boundary
    ])

    # Remove duplicate nodes
    node = np.unique(
        np.round(node, decimals=12),
        axis=0
    )

    # Triangulate the stadium
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

    return node, elm


if __name__ == "__main__":
    a = 1.0 / np.sqrt(
        4.0 + np.pi
    )

    R = a
    h = 0.10

    node, elm = stadium_mesh(
        h=h,
        a=a,
        R=R,
        n_arc=40
    )

    print("Stadium Mesh")
    print("=" * 40)
    print(f"Nodes    : {len(node)}")
    print(f"Elements : {len(elm)}")

    plt.figure(
        figsize=(9, 5)
    )

    plt.triplot(
        node[:, 0],
        node[:, 1],
        elm,
        linewidth=0.5
    )

    plt.scatter(
        node[:, 0],
        node[:, 1],
        s=5
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        f"Stadium Mesh, h = {h}"
    )

    plt.gca().set_aspect(
        "equal"
    )

    plt.tight_layout()
    plt.show()