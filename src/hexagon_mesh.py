# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:51:20 2026

@author: natoo
"""

"""
Generate a triangular FEM mesh for a regular hexagonal quantum dot.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.path import Path


def hexagon_vertices(radius=1.0, center=(0.0, 0.0)):
    """
    Return the six vertices of a regular hexagon.

    Parameters
    ----------
    radius : float
        Distance from the center to each vertex.

    center : tuple
        Coordinates of the hexagon center.

    Returns
    -------
    vertices : ndarray
        Six vertex coordinates.
    """

    xc, yc = center

    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        6,
        endpoint=False
    )

    vertices = np.column_stack((
        xc + radius * np.cos(angles),
        yc + radius * np.sin(angles)
    ))

    return vertices


def points_along_edge(p1, p2, spacing):
    """
    Generate evenly spaced points along one hexagon edge.
    """

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


def hexagon_mesh(
    spacing=0.08,
    radius=1.0,
    center=(0.0, 0.0)
):
    """
    Generate a triangular mesh inside a regular hexagon.

    Parameters
    ----------
    spacing : float
        Approximate spacing between mesh nodes.

    radius : float
        Distance from the center to each hexagon vertex.

    center : tuple
        Coordinates of the center.

    Returns
    -------
    node : ndarray
        Node coordinates.

    elm : ndarray
        Triangle connectivity.

    boundary : ndarray
        Indices of boundary nodes.
    """

    xc, yc = center

    vertices = hexagon_vertices(
        radius=radius,
        center=center
    )

    polygon = Path(vertices)

    # -------------------------------------------------
    # 1. Generate interior Cartesian grid points
    # -------------------------------------------------

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

    candidate_points = np.array([
        [x, y]
        for y in y_values
        for x in x_values
    ])

    # Keep only points strictly inside the hexagon.
    interior_mask = polygon.contains_points(
        candidate_points,
        radius=-0.2 * spacing
    )

    interior_points = candidate_points[interior_mask]

    # -------------------------------------------------
    # 2. Generate points along all six edges
    # -------------------------------------------------

    boundary_points = []

    for i in range(6):

        p1 = vertices[i]
        p2 = vertices[(i + 1) % 6]

        edge_points = points_along_edge(
            p1,
            p2,
            spacing
        )

        boundary_points.extend(edge_points)

    boundary_points = np.array(boundary_points)

    # Combine interior and boundary nodes
    node = np.vstack((
        interior_points,
        boundary_points
    ))

    first_boundary_node = len(interior_points)

    boundary = np.arange(
        first_boundary_node,
        len(node)
    )

    # -------------------------------------------------
    # 3. Delaunay triangulation
    # -------------------------------------------------

    triangulation = Delaunay(node)
    candidate_elements = triangulation.simplices.copy()

    # -------------------------------------------------
    # 4. Remove any triangles outside the hexagon
    # -------------------------------------------------

    centroids = np.mean(
        node[candidate_elements],
        axis=1
    )

    inside_elements = polygon.contains_points(
        centroids,
        radius=1e-10
    )

    elm = candidate_elements[inside_elements]

    # -------------------------------------------------
    # 5. Ensure counterclockwise node ordering
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

    node, elm, boundary = hexagon_mesh(
        spacing=0.08,
        radius=1.0
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
    plt.title("Regular Hexagonal Quantum Dot FEM Mesh")
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()