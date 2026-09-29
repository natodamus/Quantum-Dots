# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 13:06:25 2026

@author: natoo
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from square_solver import solve_square
from circle_solver import solve_circle
from hexagon_solver import solve_hexagon
from stadium_solver import solve_stadium


# Output locations
ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "results" / "figures"
DATA_DIR = ROOT / "results" / "data"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Calculate the area represented by a triangular mesh
def calculate_mesh_area(
    node,
    elm
):
    p1 = node[
        elm[:, 0]
    ]

    p2 = node[
        elm[:, 1]
    ]

    p3 = node[
        elm[:, 2]
    ]

    twice_area = (
        (p2[:, 0] - p1[:, 0])
        * (p3[:, 1] - p1[:, 1])
        - (p3[:, 0] - p1[:, 0])
        * (p2[:, 1] - p1[:, 1])
    )

    return (
        0.5
        * np.sum(
            np.abs(
                twice_area
            )
        )
    )


# Solve all four equal-area geometries using the final meshes
def solve_all_geometries(
    number_of_states=20
):
    target_area = 1.0

    # Equal-area geometry parameters
    square_side = np.sqrt(
        target_area
    )

    circle_radius = np.sqrt(
        target_area
        / np.pi
    )

    hexagon_radius = np.sqrt(
        2.0
        * target_area
        / (
            3.0
            * np.sqrt(
                3.0
            )
        )
    )

    stadium_a = np.sqrt(
        target_area
        / (
            4.0
            + np.pi
        )
    )

    stadium_R = stadium_a

    # Final resolutions selected from the convergence study
    square_N = 50
    mesh_spacing = 0.02

    square = solve_square(
        N=square_N,
        length=square_side,
        num_eigenvalues=number_of_states
    )

    circle = solve_circle(
        spacing=mesh_spacing,
        radius=circle_radius,
        n_boundary=None,
        num_eigenvalues=number_of_states
    )

    hexagon = solve_hexagon(
        spacing=mesh_spacing,
        radius=hexagon_radius,
        num_eigenvalues=number_of_states
    )

    stadium = solve_stadium(
        h=mesh_spacing,
        a=stadium_a,
        R=stadium_R,
        n_arc=80,
        num_eigenvalues=number_of_states
    )

    solutions = {
        "Square": square,
        "Circle": circle,
        "Hexagon": hexagon,
        "Stadium": stadium
    }

    results = {}

    for name, solution in solutions.items():
        (
            node,
            elm,
            boundary,
            interior,
            energies,
            eigenvectors
        ) = solution

        results[
            name
        ] = {
            "node": node,
            "elm": elm,
            "boundary": boundary,
            "interior": interior,
            "energies": energies,
            "eigenvectors": eigenvectors,
            "area": calculate_mesh_area(
                node,
                elm
            )
        }

    return results


# Print mesh sizes and numerical areas
def print_mesh_information(
    results
):
    print(
        "\nEqual-Area Mesh Information"
    )

    print(
        "-" * 72
    )

    print(
        f"{'Geometry':<12}"
        f"{'Nodes':>10}"
        f"{'Elements':>12}"
        f"{'Interior':>12}"
        f"{'Area':>14}"
        f"{'Area Error (%)':>16}"
    )

    for name, result in results.items():
        area = result[
            "area"
        ]

        area_error = (
            abs(
                area
                - 1.0
            )
            * 100.0
        )

        print(
            f"{name:<12}"
            f"{len(result['node']):>10}"
            f"{len(result['elm']):>12}"
            f"{len(result['interior']):>12}"
            f"{area:>14.6f}"
            f"{area_error:>16.4f}"
        )


# Print the energy spectrum for each geometry
def print_energy_comparison(
    results,
    number_of_states
):
    print(
        "\nEqual-Area Energy Comparison"
    )

    print(
        "-" * 79
    )

    print(
        f"{'State':>7}"
        f"{'Square':>18}"
        f"{'Circle':>18}"
        f"{'Hexagon':>18}"
        f"{'Stadium':>18}"
    )

    for state in range(
        number_of_states
    ):
        print(
            f"{state + 1:7d}"
            f"{results['Square']['energies'][state]:18.6f}"
            f"{results['Circle']['energies'][state]:18.6f}"
            f"{results['Hexagon']['energies'][state]:18.6f}"
            f"{results['Stadium']['energies'][state]:18.6f}"
        )


# Save the energy comparison for later analysis and plotting
def save_energy_comparison(
    results,
    number_of_states
):
    output_path = (
        DATA_DIR
        / "equal_area_energy_comparison.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow([
            "State",
            "Square Energy",
            "Circle Energy",
            "Hexagon Energy",
            "Stadium Energy"
        ])

        for state in range(
            number_of_states
        ):
            writer.writerow([
                state + 1,
                results[
                    "Square"
                ][
                    "energies"
                ][state],
                results[
                    "Circle"
                ][
                    "energies"
                ][state],
                results[
                    "Hexagon"
                ][
                    "energies"
                ][state],
                results[
                    "Stadium"
                ][
                    "energies"
                ][state]
            ])

    return output_path


# Plot the first several energy levels for each geometry
def plot_energy_spectra(
    results,
    number_of_states
):
    states = np.arange(
        1,
        number_of_states + 1
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    markers = {
        "Square": "o",
        "Circle": "s",
        "Hexagon": "^",
        "Stadium": "D"
    }

    for name in [
        "Square",
        "Circle",
        "Hexagon",
        "Stadium"
    ]:
        ax.plot(
            states,
            results[
                name
            ][
                "energies"
            ],
            marker=markers[
                name
            ],
            label=name
        )

    ax.set_xlabel(
        "State number"
    )

    ax.set_ylabel(
        "Energy"
    )

    ax.set_title(
        "Energy Spectra of Equal-Area Quantum Dots"
    )

    ax.set_xticks(
        states
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        FIGURES_DIR
        / "equal_area_energy_spectra.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    return output_path


# Compare the ground-state energy produced by each geometry
def plot_ground_state_comparison(
    results
):
    geometry_names = [
        "Circle",
        "Hexagon",
        "Square",
        "Stadium"
    ]

    energies = [
        results[
            name
        ][
            "energies"
        ][0]
        for name in geometry_names
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.bar(
        geometry_names,
        energies
    )

    ax.set_ylabel(
        "Ground-state energy"
    )

    ax.set_title(
        "Equal-Area Ground-State Energy Comparison"
    )

    fig.tight_layout()

    output_path = (
        FIGURES_DIR
        / "equal_area_ground_state_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    return output_path


def main():
    number_of_states = 20

    results = solve_all_geometries(
        number_of_states
    )

    print_mesh_information(
        results
    )

    print_energy_comparison(
        results,
        number_of_states
    )

    csv_path = save_energy_comparison(
        results,
        number_of_states
    )

    spectra_path = plot_energy_spectra(
        results,
        number_of_states
    )

    ground_state_path = plot_ground_state_comparison(
        results
    )

    print(
        "\nGenerated Output Files"
    )

    print(
        "-" * 45
    )

    print(
        f"Energy data           : {csv_path}"
    )

    print(
        f"Energy spectra figure : {spectra_path}"
    )

    print(
        f"Ground-state figure   : {ground_state_path}"
    )


if __name__ == "__main__":
    main()