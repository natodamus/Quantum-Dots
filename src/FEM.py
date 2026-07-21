# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:08:22 2026

@author: natoo
"""
import numpy as np

def triangle_area(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    return 0.5 * abs(
        x1 * (y2 - y3)
        + x2 * (y3 - y1)
        + x3 * (y1 - y2)
    )


def A_mat(node, elm):
    """
    Assemble stiffness matrix:
        A_ij = ∫ grad(phi_i) · grad(phi_j) dx dy
    """

    n = len(node)
    A = np.zeros((n, n))

    for e in elm:
        p1, p2, p3 = node[e]

        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        area = triangle_area(p1, p2, p3)

        beta = np.array([
            y2 - y3,
            y3 - y1,
            y1 - y2
        ])

        gamma = np.array([
            x3 - x2,
            x1 - x3,
            x2 - x1
        ])

        A_local = np.zeros((3, 3))

        for i in range(3):
            for j in range(3):
                A_local[i, j] = (
                    beta[i] * beta[j]
                    + gamma[i] * gamma[j]
                ) / (4 * area)

        for i in range(3):
            for j in range(3):
                A[e[i], e[j]] += A_local[i, j]

    return A


def B_mat(node, elm):
    """
    Assemble mass/overlap matrix:
        B_ij = ∫ phi_i phi_j dx dy
    """

    n = len(node)
    B = np.zeros((n, n))

    for e in elm:
        p1, p2, p3 = node[e]
        area = triangle_area(p1, p2, p3)

        B_local = (area / 12.0) * np.array([
            [2, 1, 1],
            [1, 2, 1],
            [1, 1, 2]
        ])

        for i in range(3):
            for j in range(3):
                B[e[i], e[j]] += B_local[i, j]

    return B