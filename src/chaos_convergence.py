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
R_GOE = 0.5307
R_GOE_SURMISE = 0.53590


# Calculate adjacent-gap ratios
def gap_ratios(energies):
    spacings = np.diff(
        np.asarray(
            energies,
            dtype=float
        )
    )

    s1 = spacings[:-1]
    s2 = spacings[1:]

    denominator = np.maximum(
        s1,
        s2
    )

    # Protect against zero or invalid spacings
    valid = (
        np.isfinite(s1)
        & np.isfinite(s2)
        & (s1 >= 0.0)
        & (s2 >= 0.0)
        & (denominator > 0.0)
    )

    return (
        np.minimum(
            s1[valid],
            s2[valid]
        )
        / denominator[valid]
    )


# Calculate the mean adjacent-gap ratio
def mean_gap_ratio(energies):
    ratios = gap_ratios(
        energies
    )

    if len(ratios) == 0:
        return np.nan

    return np.mean(
        ratios
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
        label=r"Asymptotic GOE $\langle r\rangle \approx 0.5307$"
    )

    ax.axhline(
        R_GOE_SURMISE,
        linestyle="-.",
        linewidth=1.5,
        label=r"GOE surmise $\langle r\rangle \approx 0.5359$"
    )

    ax.axhline(
        R_POISSON,
        linestyle=":",
        linewidth=1.5,
        label=r"Poisson $\langle r\rangle \approx 0.3863$"
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

    plt.close(fig)

    print(
        f"\nSaved figure to:\n{output}"
    )


# Plot sensitivity to the number of states used
def plot_level_count_stability(level_counts, level_results):
    root = Path(__file__).resolve().parent.parent
    figure_dir = root / "results" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    values = [
        level_results[n]
        for n in level_counts
    ]

    fig, ax = plt.subplots(
        figsize=(7.5, 5.0)
    )

    ax.plot(
        level_counts,
        values,
        marker="o",
        linewidth=2,
        label="Stadium FEM"
    )

    ax.axhline(
        R_GOE,
        linestyle="--",
        linewidth=1.5,
        label=r"Asymptotic GOE $\langle r\rangle \approx 0.5307$"
    )

    ax.axhline(
        R_GOE_SURMISE,
        linestyle="-.",
        linewidth=1.5,
        label=r"GOE surmise $\langle r\rangle \approx 0.5359$"
    )

    ax.axhline(
        R_POISSON,
        linestyle=":",
        linewidth=1.5,
        label=r"Poisson $\langle r\rangle \approx 0.3863$"
    )

    ax.set_xlabel(
        "Number of states per symmetry sector"
    )

    ax.set_ylabel(
        r"Combined mean gap ratio $\langle r\rangle$"
    )

    ax.set_title(
        "Gap-Ratio Stability with Number of States"
    )

    ax.set_ylim(
        0.35,
        0.60
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    output = (
        figure_dir
        / "stadium_level_count_stability.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved figure to:\n{output}"
    )


if __name__ == "__main__":
    mesh_sizes = [
        0.04,
        0.03,
        0.025,
        0.02,
        0.015,
        0.01
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
        f"Poisson <r>      = {R_POISSON:.6f}"
    )

    print(
        f"Asymptotic GOE   = {R_GOE:.6f}"
    )

    print(
        f"GOE surmise      = {R_GOE_SURMISE:.6f}"
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
                eigenvectors,
                mesh_info
            ) = solve_stadium_sector(
                sector=sector,
                h=h,
                a=a,
                R=a,
                n_arc=80,
                num_eigenvalues=number_of_states
            )

            ratios = gap_ratios(
                energies
            )

            all_ratios.append(
                ratios
            )

            results[h][sector] = {
                "energies": energies,
                "mean_r": np.mean(ratios),
                "dof": len(free_nodes),
                "mesh_info": mesh_info
            }

            print(
                f"{sector}: "
                f"DOF = {len(free_nodes):5d}, "
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

    # Relative changes between successive fine meshes
    print("\n")
    print("=" * 72)
    print("FINE-MESH RELATIVE CHANGES")
    print("=" * 72)

    fine_pairs = [
        (0.025, 0.020),
        (0.020, 0.015),
        (0.015, 0.010)
    ]

    for coarse_h, fine_h in fine_pairs:
        print(
            f"\nh = {coarse_h:.3f} -> {fine_h:.3f}"
        )

        for sector in sectors:
            coarse = results[
                coarse_h
            ][sector]["energies"]

            fine = results[
                fine_h
            ][sector]["energies"]

            changes = []

            for state in state_numbers:
                i = state - 1

                relative_change = (
                    abs(
                        fine[i] - coarse[i]
                    )
                    / fine[i]
                    * 100.0
                )

                changes.append(
                    relative_change
                )

            print(
                f"{sector}: "
                f"E1 = {changes[0]:6.3f}%  "
                f"E10 = {changes[1]:6.3f}%  "
                f"E25 = {changes[2]:6.3f}%  "
                f"E50 = {changes[3]:6.3f}%"
            )

    # Check sensitivity to the number of states
    print("\n")
    print("=" * 72)
    print("LEVEL-COUNT STABILITY")
    print("=" * 72)

    finest_h = 0.01

    level_counts = [
        10,
        20,
        30,
        40,
        50
    ]

    level_results = {}

    print(
        f"\nUsing finest mesh h = {finest_h:.3f}"
    )

    print(
        f"{'States':>10}"
        f"{'Ratios':>12}"
        f"{'Combined <r>':>18}"
    )

    for number in level_counts:
        ratios_for_number = []

        for sector in sectors:
            energies = results[
                finest_h
            ][sector]["energies"][
                :number
            ]

            ratios_for_number.append(
                gap_ratios(
                    energies
                )
            )

        combined = np.concatenate(
            ratios_for_number
        )

        level_results[number] = np.mean(
            combined
        )

        print(
            f"{number:10d}"
            f"{len(combined):12d}"
            f"{level_results[number]:18.6f}"
        )

    print("\n")
    print("=" * 72)
    print("REFERENCE")
    print("=" * 72)

    print(
        f"Poisson <r>      = {R_POISSON:.6f}"
    )

    print(
        f"Asymptotic GOE   = {R_GOE:.6f}"
    )

    print(
        f"GOE surmise      = {R_GOE_SURMISE:.6f}"
    )

    plot_chaos_convergence(
        mesh_sizes,
        results
    )

    plot_level_count_stability(
        level_counts,
        level_results
    )