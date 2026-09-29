# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:13:04 2026

@author: natoo
"""

import numpy as np

from scipy.sparse.linalg import eigsh
from scipy.special import jn_zeros

from FEM import assemble_sparse_fem
from circle_mesh import circle_mesh


# Solve the FEM eigenvalue problem for the infinite circular well
def solve_circle(
    spacing,
    radius,
    n_boundary=None,
    num_eigenvalues=50
):
    (
        node,
        elm,
        boundary
    ) = circle_mesh(
        spacing=spacing,
        radius=radius,
        center=(0.0, 0.0),
        n_boundary=n_boundary
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


# Analytical energy levels for a 2D infinite circular well
def exact_circle_energy(
    m,
    n,
    radius
):
    zero = jn_zeros(
        m,
        n
    )[-1]

    return (
        zero**2
        / (
            2.0
            * radius**2
        )
    )


if __name__ == "__main__":
    # Choose radius so the circle has area 1
    radius = (
        1.0
        / np.sqrt(
            np.pi
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
    ) = solve_circle(
        spacing=spacing,
        radius=radius,
        n_boundary=None,
        num_eigenvalues=50
    )

    # Compare the numerical ground state with the analytical solution
    exact_ground = exact_circle_energy(
        0,
        1,
        radius
    )

    error = (
        abs(
            energies[0]
            - exact_ground
        )
        / exact_ground
        * 100.0
    )

    print("Circular Quantum Dot")
    print("-" * 54)

    print(
        f"Radius            : {radius:.10f}"
    )

    print(
        f"Exact area        : "
        f"{np.pi * radius**2:.10f}"
    )

    print(
        f"Mesh spacing      : {spacing}"
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