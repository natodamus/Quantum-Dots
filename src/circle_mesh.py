# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:08:01 2026

@author: natoo
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


def circle_mesh(
    spacing=0.08,
    radius=1.0,
    center=(0.0, 0.0),
    n_boundary=100
):
    

    xc, yc = center

    # -------------------------------------------------
    # 1. Generate interior Cartesian grid points
    # -------------------------------------------------
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

    interior_points = []

    for y in y_values:
        for x in x_values:

            distance_squared = (x - xc)**2 + (y - yc)**2

            # Keep points strictly inside the circle.
            # Leave a small gap so they do not duplicate
            # the separately generated boundary nodes.
            if distance_squared < (radius - 0.25 * spacing)**2:
                interior_points.append([x, y])

    interior_points = np.array(interior_points)

    # -------------------------------------------------
    # 2. Generate circular boundary points
    # -------------------------------------------------
    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        n_boundary,
        endpoint=False
    )

    boundary_points = np.column_stack((
        xc + radius * np.cos(theta),
        yc + radius * np.sin(theta)
    ))

    # Combine interior and boundary nodes
    node = np.vstack((interior_points, boundary_points))

    # Boundary nodes were added last
    first_boundary_node = len(interior_points)

    boundary = np.arange(
        first_boundary_node,
        len(node)
    )

    # -------------------------------------------------
    # 3. Construct triangular elements
    # -------------------------------------------------
    triangulation = Delaunay(node)
    elm = triangulation.simplices.copy()

    # -------------------------------------------------
    # 4. Ensure all triangles are counterclockwise
    # -------------------------------------------------
    for k, triangle in enumerate(elm):

        p1 = node[triangle[0]]
        p2 = node[triangle[1]]
        p3 = node[triangle[2]]

        signed_area = 0.5 * (
            (p2[0] - p1[0]) * (p3[1] - p1[1])
            - (p3[0] - p1[0]) * (p2[1] - p1[1])
        )

        if signed_area < 0:
            elm[k, [1, 2]] = elm[k, [2, 1]]

    return node, elm, boundary


# =====================================================
# Mesh visualization test
# =====================================================
if __name__ == "__main__":

    node, elm, boundary = circle_mesh(
        spacing=0.08,
        radius=1.0,
        n_boundary=100
    )

    print("Number of nodes:", len(node))
    print("Number of elements:", len(elm))
    print("Number of boundary nodes:", len(boundary))

    plt.figure(figsize=(7, 7))

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
    plt.title("Circular Quantum Dot FEM Mesh")
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()