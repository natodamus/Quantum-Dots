# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:44:47 2026

@author: natoo
"""

import numpy as np

from stadium_sector_solver import solve_stadium_sector

import matplotlib.pyplot as plt
from pathlib import Path

# Reference values for adjacent-gap statistics
R_POISSON = 2.0 * np.log(2.0) - 1.0
R_GOE = 0.53590


# Calculate adjacent-gap ratios
def gap_ratios(energies):
    spacings = np.diff(energies)

    return np.minimum(
        spacings[:-1],
        spacings[1:]
    ) / np.maximum(
        spacings[:-1],
        spacings[1:]
    )


def mean_gap_ratio(energies):
    return np.mean(
        gap_ratios(energies)
    )

# Plot convergence of the combined gap-ratio statistic
def plot_chaos_convergence(mesh_sizes, results):
    root = Path(__file__).resolve().parent.parent
    figure_dir = root / "results" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    combined = [
        results[h]["combined"]
        for h in mesh_sizes
    ]

    fig, ax = plt.subplots(
        figsize=(7.5, 5.0)
    )

    ax.plot(
        mesh_sizes,
        combined,
        marker="o",
        linewidth=2,
        label="Stadium FEM"
    )

    ax.axhline(
        R_GOE,
        linestyle="--",
        linewidth=1.5,
        label=r"GOE $\langle r\rangle = 0.5359$"
    )

    ax.axhline(
        R_POISSON,
        linestyle=":",
        linewidth=1.5,
        label=r"Poisson $\langle r\rangle = 0.3863$"
    )

    ax.set_xlabel(
        "Mesh size h"
    )

    ax.set_ylabel(
        r"Combined mean gap ratio $\langle r\rangle$"
    )

    ax.set_title(
        "Stadium Quantum-Chaos Convergence"
    )

    # Finer meshes appear toward the right
    ax.invert_xaxis()

    ax.set_ylim(
        0.35,
        0.59
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    output = (
        figure_dir
        / "stadium_chaos_convergence.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"\nSaved figure to:\n{output}"
    )

if __name__ == "__main__":
    mesh_sizes = [
        0.04,
        0.03,
        0.025,
        0.02
    ]

    sectors = [
        "++",
        "+-",
        "-+",
        "--"
    ]

    number_of_states = 50

    a = 1.0 / np.sqrt(
        4.0 + np.pi
    )

    results = {}

    print("Stadium Quantum-Chaos Convergence")
    print("=" * 72)

    print(
        f"Poisson <r> = {R_POISSON:.6f}"
    )

    print(
        f"GOE <r>     = {R_GOE:.6f}"
    )

    # Run each mesh resolution
    for h in mesh_sizes:
        print(
            f"\nMesh h = {h:.3f}"
        )

        print("-" * 72)

        results[h] = {}

        all_ratios = []

        for sector in sectors:
            (
                node,
                elm,
                dirichlet,
                free_nodes,
                energies,
                eigenvectors
            ) = solve_stadium_sector(
                sector=sector,
                h=h,
                a=a,
                R=a,
                n_arc=80
            )

            energies = energies[
                :number_of_states
            ]

            ratios = gap_ratios(
                energies
            )

            all_ratios.append(
                ratios
            )

            results[h][sector] = {
                "energies": energies,
                "mean_r": np.mean(ratios)
            }

            print(
                f"{sector}: "
                f"DOF = {len(free_nodes):4d}, "
                f"<r> = {np.mean(ratios):.6f}"
            )

        combined = np.concatenate(
            all_ratios
        )

        results[h]["combined"] = np.mean(
            combined
        )

        print(
            f"Combined <r> = "
            f"{results[h]['combined']:.6f}"
        )

    # Summary of gap-ratio convergence
    print("\n")
    print("=" * 72)
    print("GAP-RATIO CONVERGENCE")
    print("=" * 72)

    print(
        f"{'h':>8}"
        f"{'++':>12}"
        f"{'+-':>12}"
        f"{'-+':>12}"
        f"{'--':>12}"
        f"{'Combined':>12}"
    )

    for h in mesh_sizes:
        print(
            f"{h:8.3f}"
            f"{results[h]['++']['mean_r']:12.6f}"
            f"{results[h]['+-']['mean_r']:12.6f}"
            f"{results[h]['-+']['mean_r']:12.6f}"
            f"{results[h]['--']['mean_r']:12.6f}"
            f"{results[h]['combined']:12.6f}"
        )

    # Check convergence of representative eigenvalues
    print("\n")
    print("=" * 72)
    print("SELECTED EIGENVALUE CONVERGENCE")
    print("=" * 72)

    state_numbers = [
        1,
        10,
        25,
        50
    ]

    for sector in sectors:
        print(
            f"\nSector {sector}"
        )

        print(
            f"{'h':>8}"
            f"{'E1':>14}"
            f"{'E10':>14}"
            f"{'E25':>14}"
            f"{'E50':>14}"
        )

        for h in mesh_sizes:
            energies = results[h][
                sector
            ]["energies"]

            values = [
                energies[state - 1]
                for state in state_numbers
            ]

            print(
                f"{h:8.3f}"
                f"{values[0]:14.6f}"
                f"{values[1]:14.6f}"
                f"{values[2]:14.6f}"
                f"{values[3]:14.6f}"
            )

    # Relative change from the two finest meshes
    print("\n")
    print("=" * 72)
    print("FINE-MESH RELATIVE CHANGES")
    print("=" * 72)

    coarse_h = 0.025
    fine_h = 0.020

    for sector in sectors:
        coarse = results[
            coarse_h
        ][sector]["energies"]

        fine = results[
            fine_h
        ][sector]["energies"]

        print(
            f"\nSector {sector}"
        )

        for state in state_numbers:
            i = state - 1

            relative_change = (
                abs(
                    fine[i] - coarse[i]
                )
                / fine[i]
                * 100.0
            )

            print(
                f"E{state:<2d}: "
                f"{relative_change:.4f}%"
            )

    print("\n")
    print("=" * 72)
    print("REFERENCE")
    print("=" * 72)

    print(
        f"Poisson <r> = {R_POISSON:.6f}"
    )

    print(
        f"GOE <r>     = {R_GOE:.6f}"
    )
    
    plot_chaos_convergence(
    mesh_sizes,
    results
)