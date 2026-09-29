# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:47:17 2026

@author: natoo
"""
import numpy as np

from collections import Counter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from stadium_mesh import stadium_mesh


# Find boundary nodes from edges that belong to only one triangle
def find_boundary_nodes(elm):
    edges = []

    for triangle in elm:
        n0, n1, n2 = triangle

        edges.append(
            tuple(
                sorted([
                    n0,
                    n1
                ])
            )
        )

        edges.append(
            tuple(
                sorted([
                    n1,
                    n2
                ])
            )
        )

        edges.append(
            tuple(
                sorted([
                    n2,
                    n0
                ])
            )
        )

    edge_counts = Counter(
        edges
    )

    boundary_edges = [
        edge
        for edge, count in edge_counts.items()
        if count == 1
    ]

    return np.unique(
        np.asarray(
            boundary_edges,
            dtype=int
        ).ravel()
    )


# Assemble sparse FEM stiffness and mass matrices
def assemble_sparse_fem(node, elm):
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
            * abs(twice_area)
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

    return A, B


# Solve the FEM eigenvalue problem for the infinite stadium-shaped well
def solve_stadium(
    h,
    a=0.5,
    R=0.5,
    n_arc=80,
    num_eigenvalues=50
):
    node, elm = stadium_mesh(
        h=h,
        a=a,
        R=R,
        n_arc=n_arc
    )

    # Apply psi = 0 on the physical boundary
    boundary = find_boundary_nodes(
        elm
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

    # Number of requested eigenvalues cannot exceed matrix size
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

    # Sort eigenvalues and eigenvectors in ascending energy
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
    # Use a = R so the full stadium has area 1
    a = 1.0 / np.sqrt(
        4.0 + np.pi
    )

    R = a
    h = 0.02

    (
        node,
        elm,
        boundary,
        interior,
        energies,
        eigenvectors
    ) = solve_stadium(
        h=h,
        a=a,
        R=R,
        n_arc=80,
        num_eigenvalues=50
    )

    # Stadium area = rectangle + two semicircular end caps
    area = (
        4.0
        * a
        * R
        + np.pi
        * R**2
    )

    print("Stadium Quantum Dot")
    print("-" * 54)

    print(
        f"a            : {a:.6f}"
    )

    print(
        f"R            : {R:.6f}"
    )

    print(
        f"Exact area   : {area:.10f}"
    )

    print(
        f"Mesh spacing : {h}"
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