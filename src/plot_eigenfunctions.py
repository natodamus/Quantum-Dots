# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:49:25 2026

@author: natoo
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

from square_solver import solve_square
from circle_solver import solve_circle
from hexagon_solver import solve_hexagon
from stadium_solver import solve_stadium


# Output location
ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Solve the four equal-area geometries using the final mesh resolutions
def solve_geometries(number_of_states=6):
    square_side = 1.0
    circle_radius = 1.0 / np.sqrt(np.pi)
    hexagon_radius = np.sqrt(2.0 / (3.0 * np.sqrt(3.0)))

    stadium_a = 1.0 / np.sqrt(4.0 + np.pi)
    stadium_R = stadium_a

    square = solve_square(
        N=50,
        length=square_side
    )

    circle = solve_circle(
        spacing=0.03,
        radius=circle_radius,
        n_boundary=100
    )

    hexagon = solve_hexagon(
        spacing=0.03,
        radius=hexagon_radius
    )

    stadium = solve_stadium(
        h=0.03,
        a=stadium_a,
        R=stadium_R,
        n_arc=60
    )

    solutions = {
        "Square": square,
        "Circle": circle,
        "Hexagon": hexagon,
        "Stadium": stadium
    }

    results = {}

    for name, solution in solutions.items():
        node, elm, boundary, interior, energies, eigenvectors = solution

        results[name] = {
            "node": node,
            "elm": elm,
            "boundary": boundary,
            "interior": interior,
            "energies": energies[:number_of_states],
            "eigenvectors": eigenvectors[:, :number_of_states]
        }

    return results


# Reconstruct an eigenfunction on the full mesh including boundary nodes
def reconstruct_wavefunction(result, state):
    node = result["node"]
    interior = result["interior"]
    eigenvectors = result["eigenvectors"]

    psi = np.zeros(len(node))
    psi[interior] = eigenvectors[:, state]

    # Normalize the amplitude for consistent plotting
    psi /= np.max(np.abs(psi))

    return psi


# Plot selected wavefunctions for all four geometries
def plot_wavefunctions(results, states):
    geometry_names = ["Square", "Circle", "Hexagon", "Stadium"]

    fig, axes = plt.subplots(
        len(geometry_names),
        len(states),
        figsize=(4 * len(states), 3.5 * len(geometry_names)),
        constrained_layout=True
    )

    for row, name in enumerate(geometry_names):
        result = results[name]
        node = result["node"]
        elm = result["elm"]

        triangulation = tri.Triangulation(
            node[:, 0],
            node[:, 1],
            elm
        )

        for col, state in enumerate(states):
            ax = axes[row, col]

            psi = reconstruct_wavefunction(result, state)

            contour = ax.tricontourf(
                triangulation,
                psi,
                levels=40,
                cmap="RdBu_r",
                vmin=-1.0,
                vmax=1.0
            )

            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

            energy = result["energies"][state]

            ax.set_title(
                f"State {state + 1}\nE = {energy:.3f}"
            )

            if col == 0:
                ax.set_ylabel(
                    name,
                    fontsize=12,
                    fontweight="bold"
                )

    colorbar = fig.colorbar(
        contour,
        ax=axes,
        shrink=0.7,
        pad=0.02
    )

    colorbar.set_label(r"Normalized $\psi$")

    fig.suptitle(
        "Eigenfunctions of Equal-Area Quantum Dots",
        fontsize=16
    )

    output_path = FIGURES_DIR / "eigenfunction_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


# Plot probability densities for the same selected states
def plot_probability_densities(results, states):
    geometry_names = ["Square", "Circle", "Hexagon", "Stadium"]

    fig, axes = plt.subplots(
        len(geometry_names),
        len(states),
        figsize=(4 * len(states), 3.5 * len(geometry_names)),
        constrained_layout=True
    )

    for row, name in enumerate(geometry_names):
        result = results[name]
        node = result["node"]
        elm = result["elm"]

        triangulation = tri.Triangulation(
            node[:, 0],
            node[:, 1],
            elm
        )

        for col, state in enumerate(states):
            ax = axes[row, col]

            psi = reconstruct_wavefunction(result, state)
            probability = np.abs(psi)**2

            contour = ax.tricontourf(
                triangulation,
                probability,
                levels=40,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0
            )

            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

            energy = result["energies"][state]

            ax.set_title(
                f"State {state + 1}\nE = {energy:.3f}"
            )

            if col == 0:
                ax.set_ylabel(
                    name,
                    fontsize=12,
                    fontweight="bold"
                )

    colorbar = fig.colorbar(
        contour,
        ax=axes,
        shrink=0.7,
        pad=0.02
    )

    colorbar.set_label(r"Normalized $|\psi|^2$")

    fig.suptitle(
        "Probability Densities of Equal-Area Quantum Dots",
        fontsize=16
    )

    output_path = FIGURES_DIR / "probability_density_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def main():
    # States 1-3 show the ground state and the first excited-state structure
    states = [0, 1, 2]

    results = solve_geometries(
        number_of_states=max(states) + 1
    )

    wavefunction_path = plot_wavefunctions(
        results,
        states
    )

    probability_path = plot_probability_densities(
        results,
        states
    )

    print("Eigenfunction Figures")
    print("-" * 45)
    print(f"Wavefunctions        : {wavefunction_path}")
    print(f"Probability densities: {probability_path}")


if __name__ == "__main__":
    main()