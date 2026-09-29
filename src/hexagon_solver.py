# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:52:20 2026

@author: natoo
"""
import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from hexagon_mesh import hexagon_mesh


# Assemble sparse FEM stiffness and mass matrices
def assemble_sparse_fem(
    node,
    elm
):
    rows = []
    cols = []
    stiffness_data = []
    mass_data = []

    local_mass_pattern = np.array([
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0]
    ])

    for triangle in elm:
        coordinates = node[
            triangle
        ]

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        twice_area = (
            (x[1] - x[0])
            * (y[2] - y[0])
            - (x[2] - x[0])
            * (y[1] - y[0])
        )

        area = (
            0.5
            * abs(
                twice_area
            )
        )

        if area <= 0.0:
            raise ValueError(
                "Mesh contains a zero-area triangle."
            )

        b = np.array([
            y[1] - y[2],
            y[2] - y[0],
            y[0] - y[1]
        ])

        c = np.array([
            x[2] - x[1],
            x[0] - x[2],
            x[1] - x[0]
        ])

        local_stiffness = (
            np.outer(
                b,
                b
            )
            + np.outer(
                c,
                c
            )
        ) / (
            4.0
            * area
        )

        local_mass = (
            area
            / 12.0
            * local_mass_pattern
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
                    local_stiffness[i, j]
                )

                mass_data.append(
                    local_mass[i, j]
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


# Solve the FEM eigenvalue problem for the infinite hexagonal well
def solve_hexagon(
    spacing,
    radius,
    num_eigenvalues=50
):
    (
        node,
        elm,
        boundary
    ) = hexagon_mesh(
        spacing=spacing,
        radius=radius,
        center=(0.0, 0.0)
    )

    # Interior nodes are the free FEM degrees of freedom
    interior = np.setdiff1d(
        np.arange(
            len(node)
        ),
        boundary
    )

    # Assemble sparse FEM stiffness and mass matrices
    A, B = assemble_sparse_fem(
        node,
        elm
    )

    A_int = A[
        interior
    ][
        :,
        interior
    ]

    B_int = B[
        interior
    ][
        :,
        interior
    ]

    # Check that every free node carries positive mass
    mass_diagonal = B_int.diagonal()

    if np.any(
        mass_diagonal <= 0.0
    ):
        raise ValueError(
            "Mass matrix contains zero or negative diagonal entries."
        )

    # Limit the requested spectrum to the available matrix size
    k = min(
        num_eigenvalues,
        len(interior) - 2
    )

    if k < 1:
        raise ValueError(
            "Mesh does not contain enough interior nodes."
        )

    # Solve (1/2)A psi = E B psi near the bottom of the spectrum
    energies, eigenvectors = eigsh(
        0.5 * A_int,
        k=k,
        M=B_int,
        sigma=0.0,
        which="LM"
    )

    # Sort the eigenpairs in ascending energy
    order = np.argsort(
        energies
    )

    energies = energies[
        order
    ]

    eigenvectors = eigenvectors[
        :,
        order
    ]

    return (
        node,
        elm,
        boundary,
        interior,
        energies,
        eigenvectors
    )


if __name__ == "__main__":
    # Choose the radius so the regular hexagon has area 1
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
        boundary,
        interior,
        energies,
        eigenvectors
    ) = solve_hexagon(
        spacing=spacing,
        radius=radius,
        num_eigenvalues=50
    )

    # Exact area of the regular hexagon
    area = (
        3.0
        * np.sqrt(3.0)
        / 2.0
        * radius**2
    )

    print("Hexagonal Quantum Dot")
    print("-" * 54)

    print(
        f"Radius       : {radius:.10f}"
    )

    print(
        f"Exact area   : {area:.10f}"
    )

    print(
        f"Mesh spacing : {spacing}"
    )

    print(
        f"Nodes        : {len(node)}"
    )

    print(
        f"Elements     : {len(elm)}"
    )

    print(
        f"Boundary     : {len(boundary)}"
    )

    print(
        f"Interior DOF : {len(interior)}"
    )

    print(
        "\nFirst 10 FEM energies"
    )

    for state, energy in enumerate(
        energies[:10],
        start=1
    ):
        print(
            f"State {state:2d}: "
            f"E = {energy:.8f}"
        )
