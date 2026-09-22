# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:13:15 2026

@author: natoo
"""

import numpy as np
from scipy.special import jn_zeros

from square_solver import solve_square
from circle_solver import solve_circle
from hexagon_solver import solve_hexagon
from stadium_solver import solve_stadium
from hexagon_basis import solve_hexagon_basis
from stadium_basis import solve_stadium_basis


# Print the difference between FEM and a reference spectrum
def compare(name, fem, reference, reference_name, n=10):
    print(f"\n{name}: FEM vs {reference_name}")
    print("-" * 58)

    print(
        f"{'State':>8}"
        f"{'FEM':>14}"
        f"{reference_name:>14}"
        f"{'Diff (%)':>12}"
    )

    for i in range(min(n, len(fem), len(reference))):
        difference = abs(fem[i] - reference[i]) / abs(fem[i]) * 100.0

        print(
            f"{i + 1:8d}"
            f"{fem[i]:14.6f}"
            f"{reference[i]:14.6f}"
            f"{difference:12.3f}"
        )


# Generate the analytical spectrum of a unit square
def exact_square_spectrum(number_of_states=10):
    levels = []

    for nx in range(1, 15):
        for ny in range(1, 15):
            energy = 0.5 * np.pi**2 * (nx**2 + ny**2)
            levels.append(energy)

    return np.sort(np.asarray(levels))[:number_of_states]


# Generate the analytical spectrum of an equal-area circle
def exact_circle_spectrum(radius, number_of_states=10):
    levels = []

    for m in range(21):
        zeros = jn_zeros(m, 20)

        for zero in zeros:
            energy = zero**2 / (2.0 * radius**2)
            levels.append(energy)

            # Angular states with m > 0 are twofold degenerate
            if m > 0:
                levels.append(energy)

    return np.sort(np.asarray(levels))[:number_of_states]


if __name__ == "__main__":
    number_of_states = 10

    # Use the final FEM resolutions selected from the convergence study
    square_fem = solve_square(N=50)[4]

    circle_radius = 1.0 / np.sqrt(np.pi)
    circle_fem = solve_circle(
        spacing=0.03,
        radius=circle_radius,
        n_boundary=100
    )[4]

    hexagon_radius = np.sqrt(2.0 / (3.0 * np.sqrt(3.0)))
    hexagon_fem = solve_hexagon(
        spacing=0.03,
        radius=hexagon_radius
    )[4]

    stadium_a = 1.0 / np.sqrt(4.0 + np.pi)
    stadium_fem = solve_stadium(
        h=0.03,
        a=stadium_a,
        R=stadium_a,
        n_arc=60
    )[4]

    # Use exact spectra for the analytically solvable geometries
    square_exact = exact_square_spectrum(number_of_states)

    circle_exact = exact_circle_spectrum(
        circle_radius,
        number_of_states
    )

    # Use the finite-barrier basis as an independent numerical comparison
    hexagon_basis = solve_hexagon_basis(
        M=18,
        grid_size=260,
        V0=1e4,
        num_states=number_of_states
    )[0]

    stadium_basis = solve_stadium_basis(
        M=18,
        grid_size=260,
        V0=1e4,
        num_states=number_of_states
    )[0]

    print("\n" + "=" * 62)
    print("QUANTUM DOT METHOD COMPARISON")
    print("=" * 62)

    compare(
        "Square",
        square_fem,
        square_exact,
        "Exact",
        number_of_states
    )

    compare(
        "Circle",
        circle_fem,
        circle_exact,
        "Exact",
        number_of_states
    )

    compare(
        "Hexagon",
        hexagon_fem,
        hexagon_basis,
        "Basis",
        number_of_states
    )

    compare(
        "Stadium",
        stadium_fem,
        stadium_basis,
        "Basis",
        number_of_states
    )

    print("\nBasis-method note:")
    print(
        "Hexagon and stadium use a finite V0 = 1e4 barrier with an "
        "18 x 18 rectangular sine basis."
    )
    print(
        "The basis results provide an independent numerical comparison "
        "but use a finite barrier and truncated basis."
    )
    print(
        "They should not be interpreted as exact infinite-wall "
        "reference energies."
    )