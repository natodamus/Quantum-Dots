# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:08:22 2026

@author: natoo
"""
import numpy as np

from scipy.sparse import coo_matrix


# Calculate the area of one triangular element
def triangle_area(
    p1,
    p2,
    p3
):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    return (
        0.5
        * abs(
            x1 * (y2 - y3)
            + x2 * (y3 - y1)
            + x3 * (y1 - y2)
        )
    )


# Calculate the local stiffness matrix for one P1 triangle
def local_stiffness_matrix(
    coordinates
):
    x = coordinates[:, 0]
    y = coordinates[:, 1]

    area = triangle_area(
        coordinates[0],
        coordinates[1],
        coordinates[2]
    )

    if area <= 0.0:
        raise ValueError(
            "Mesh contains a zero-area triangle."
        )

    beta = np.array([
        y[1] - y[2],
        y[2] - y[0],
        y[0] - y[1]
    ])

    gamma = np.array([
        x[2] - x[1],
        x[0] - x[2],
        x[1] - x[0]
    ])

    return (
        np.outer(
            beta,
            beta
        )
        + np.outer(
            gamma,
            gamma
        )
    ) / (
        4.0
        * area
    )


# Calculate the local mass matrix for one P1 triangle
def local_mass_matrix(
    coordinates
):
    area = triangle_area(
        coordinates[0],
        coordinates[1],
        coordinates[2]
    )

    if area <= 0.0:
        raise ValueError(
            "Mesh contains a zero-area triangle."
        )

    return (
        area
        / 12.0
        * np.array([
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0]
        ])
    )


# Assemble the dense FEM stiffness matrix
def A_mat(
    node,
    elm
):
    n = len(
        node
    )

    A = np.zeros(
        (
            n,
            n
        )
    )

    for triangle in elm:
        coordinates = node[
            triangle
        ]

        A_local = local_stiffness_matrix(
            coordinates
        )

        for i in range(3):
            for j in range(3):
                A[
                    triangle[i],
                    triangle[j]
                ] += A_local[
                    i,
                    j
                ]

    return A


# Assemble the dense FEM mass matrix
def B_mat(
    node,
    elm
):
    n = len(
        node
    )

    B = np.zeros(
        (
            n,
            n
        )
    )

    for triangle in elm:
        coordinates = node[
            triangle
        ]

        B_local = local_mass_matrix(
            coordinates
        )

        for i in range(3):
            for j in range(3):
                B[
                    triangle[i],
                    triangle[j]
                ] += B_local[
                    i,
                    j
                ]

    return B


# Assemble sparse FEM stiffness and mass matrices
def assemble_sparse_fem(
    node,
    elm
):
    rows = []
    cols = []

    stiffness_data = []
    mass_data = []

    for triangle in elm:
        coordinates = node[
            triangle
        ]

        A_local = local_stiffness_matrix(
            coordinates
        )

        B_local = local_mass_matrix(
            coordinates
        )

        for i in range(3):
            for j in range(3):
                rows.append(
                    triangle[i]
                )

                cols.append(
                    triangle[j]
                )

                stiffness_data.append(
                    A_local[
                        i,
                        j
                    ]
                )

                mass_data.append(
                    B_local[
                        i,
                        j
                    ]
                )

    shape = (
        len(node),
        len(node)
    )

    A = coo_matrix(
        (
            stiffness_data,
            (
                rows,
                cols
            )
        ),
        shape=shape
    ).tocsr()

    B = coo_matrix(
        (
            mass_data,
            (
                rows,
                cols
            )
        ),
        shape=shape
    ).tocsr()

    return (
        A,
        B
    )


if __name__ == "__main__":
    # Simple unit right triangle for an assembly check
    node = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    elm = np.array([
        [0, 1, 2]
    ])

    A_dense = A_mat(
        node,
        elm
    )

    B_dense = B_mat(
        node,
        elm
    )

    A_sparse, B_sparse = assemble_sparse_fem(
        node,
        elm
    )

    stiffness_difference = np.max(
        np.abs(
            A_dense
            - A_sparse.toarray()
        )
    )

    mass_difference = np.max(
        np.abs(
            B_dense
            - B_sparse.toarray()
        )
    )

    print("FEM Assembly Check")
    print("=" * 54)

    print(
        f"Triangle area              : "
        f"{triangle_area(node[0], node[1], node[2]):.6f}"
    )

    print(
        f"Maximum stiffness difference: "
        f"{stiffness_difference:.3e}"
    )

    print(
        f"Maximum mass difference     : "
        f"{mass_difference:.3e}"
    )