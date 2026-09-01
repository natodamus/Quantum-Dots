# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:43:31 2026

@author: natoo
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


def inside_stadium(x, y, a, R):
    """
    Check whether a point lies inside the stadium.

    Stadium consists of:
    - a central rectangle from x = -a to x = a
    - a left semicircle of radius R
    - a right semicircle of radius R
    """

    # Central rectangle
    if -a <= x <= a and -R <= y <= R:
        return True

    # Left semicircle
    if x < -a:
        return (x + a)**2 + y**2 <= R**2

    # Right semicircle
    if x > a:
        return (x - a)**2 + y**2 <= R**2

    return False


def stadium_mesh(h=0.10, a=0.5, R=0.5, n_arc=40):
    """
    Generate nodes and triangular elements for a stadium.

    h:
        approximate spacing between interior points

    a:
        half-length of central rectangle

    R:
        radius of semicircular ends

    n_arc:
        number of points used on each semicircle
    """

    points = []

    # --------------------------------------------------
    # 1. Interior points
    # --------------------------------------------------

    xmin = -(a + R)
    xmax = +(a + R)

    ymin = -R
    ymax = +R

    xs = np.arange(xmin, xmax + h, h)
    ys = np.arange(ymin, ymax + h, h)

    for y in ys:
        for x in xs:

            if inside_stadium(x, y, a, R):
                points.append([x, y])


    # --------------------------------------------------
    # 2. Straight parts of boundary
    # --------------------------------------------------

    n_line = int((2 * a) / h) + 1

    x_line = np.linspace(
        -a,
        a,
        n_line
    )

    for x in x_line:

        # top boundary
        points.append([x, R])

        # bottom boundary
        points.append([x, -R])


    # --------------------------------------------------
    # 3. Right semicircle
    # --------------------------------------------------

    theta = np.linspace(
        -np.pi / 2,
        np.pi / 2,
        n_arc
    )

    for t in theta:

        x = a + R * np.cos(t)
        y = R * np.sin(t)

        points.append([x, y])


    # --------------------------------------------------
    # 4. Left semicircle
    # --------------------------------------------------

    theta = np.linspace(
        np.pi / 2,
        3 * np.pi / 2,
        n_arc
    )

    for t in theta:

        x = -a + R * np.cos(t)
        y = R * np.sin(t)

        points.append([x, y])


    # --------------------------------------------------
    # 5. Remove duplicate nodes
    # --------------------------------------------------

    node = np.array(points)

    node = np.unique(
        np.round(node, 12),
        axis=0
    )


    # --------------------------------------------------
    # 6. Triangulate
    # --------------------------------------------------

    triangulation = Delaunay(node)

    elm = triangulation.simplices

    return node, elm


# ======================================================
# Create mesh
# ======================================================

if __name__ == "__main__":

    a = 1 / np.sqrt(4 + np.pi)
    R = a
    h = 0.10

    node, elm = stadium_mesh(
        h=h,
        a=a,
        R=R,
        n_arc=40
    )

    print("Number of nodes:", len(node))
    print("Number of triangles:", len(elm))

    plt.figure(figsize=(9, 5))

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
    plt.title(f"Stadium Mesh, h = {h}")
    plt.gca().set_aspect("equal")

    plt.tight_layout()
    plt.show()