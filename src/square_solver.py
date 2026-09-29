# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:39:01 2026

@author: natoo
"""

import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


# Generate a triangular mesh for a square domain
def square_mesh(
    N,
    length=1.0
):
    node = []

    for j in range(N + 1):
        for i in range(N + 1):
            node.append([
                length * i / N,
                length * j / N
            ])

    node = np.asarray(
        node,
        dtype=float
    )

    elm = []

    # Divide each square cell into two triangular elements
    for j in range(N):
        for i in range(N):
            n0 = (
                j * (N + 1)
                + i
            )

            n1 = (
                n0 + 1
            )

            n2 = (
                n0 + (N + 1)
            )

            n3 = (
                n2 + 1
            )

            elm.append([
                n0,
                n1,
                n3
            ])

            elm.append([
                n0,
                n3,
                n2
            ])

    return (
        node,
        np.asarray(
            elm,
            dtype=int
        )
    )


# Identify nodes along the boundary of the square
def square_boundary_nodes(
    node,
    length=1.0
):
    x = node[:, 0]
    y = node[:, 1]

    boundary_mask = (
        np.isclose(
            x,
            0.0
        )
        | np.isclose(
            x,
            length
        )
        | np.isclose(
            y,
            0.0
        )
        | np.isclose(
            y,
            length
        )
    )

    return np.where(
        boundary_mask
    )[0]


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


# Solve the FEM eigenvalue problem for the infinite square well
def solve_square(
    N,
    length=1.0,
    num_eigenvalues=50
):
    node, elm = square_mesh(
        N,
        length
    )

    # Identify fixed and free nodes
    boundary = square_boundary_nodes(
        node,
        length
    )

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


# Analytical energy levels for a 2D infinite square well
def exact_square_energy(
    nx,
    ny,
    length=1.0
):
    return (
        np.pi**2
        / (
            2.0
            * length**2
        )
        * (
            nx**2
            + ny**2
        )
    )


if __name__ == "__main__":
    N = 50
    length = 1.0

    (
        node,
        elm,
        boundary,
        interior,
        energies,
        eigenvectors
    ) = solve_square(
        N=N,
        length=length,
        num_eigenvalues=50
    )

    # Compare with E_11 = pi^2 for the unit square
    exact_ground = exact_square_energy(
        1,
        1,
        length
    )

    error = (
        abs(
            energies[0]
            - exact_ground
        )
        / exact_ground
        * 100.0
    )

    print("Square Quantum Dot")
    print("-" * 54)

    print(
        f"Length            : {length:.6f}"
    )

    print(
        f"Exact area        : {length**2:.6f}"
    )

    print(
        f"Mesh subdivisions : {N}"
    )

    print(
        f"Mesh spacing      : {length / N:.6f}"
    )

    print(
        f"Nodes             : {len(node)}"
    )

    print(
        f"Elements          : {len(elm)}"
    )

    print(
        f"Boundary          : {len(boundary)}"
    )

    print(
        f"Interior DOF      : {len(interior)}"
    )

    print(
        f"FEM ground energy : "
        f"{energies[0]:.8f}"
    )

    print(
        f"Exact energy      : "
        f"{exact_ground:.8f}"
    )

    print(
        f"Relative error    : "
        f"{error:.6f}%"
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