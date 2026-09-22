# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:32:21 2026

@author: natoo
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial import cKDTree

from stadium_solver import solve_stadium


# Set output directories
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "results" / "data"
FIGURES_DIR = ROOT / "results" / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Map reflected coordinates to the nearest nodes in the mesh
def reflection_map(node, axis):
    tree = cKDTree(node)

    reflected = node.copy()

    if axis == "x":
        reflected[:, 0] *= -1.0

    elif axis == "y":
        reflected[:, 1] *= -1.0

    else:
        raise ValueError("axis must be 'x' or 'y'")

    distances, indices = tree.query(reflected)

    return indices, distances


# Reconstruct an eigenfunction on the full mesh
def full_eigenfunction(eigenvector, interior, number_of_nodes):
    psi = np.zeros(number_of_nodes)
    psi[interior] = eigenvector

    return psi


# Measure the parity of an eigenfunction under reflection
def parity_correlation(psi, reflection_indices):
    reflected_psi = psi[reflection_indices]

    denominator = np.dot(psi, psi)

    if denominator == 0.0:
        return 0.0

    return np.dot(
        psi,
        reflected_psi
    ) / denominator


# Convert parity correlations into a symmetry label
def classify_sector(px, py, threshold=0.8):
    if abs(px) < threshold or abs(py) < threshold:
        return "mixed"

    x_label = "+" if px > 0.0 else "-"
    y_label = "+" if py > 0.0 else "-"

    return x_label + y_label


# Calculate adjacent-gap ratios within one symmetry sector
def adjacent_gap_ratios(energies):
    if len(energies) < 3:
        return np.asarray([])

    spacings = np.diff(energies)

    return np.minimum(
        spacings[:-1],
        spacings[1:]
    ) / np.maximum(
        spacings[:-1],
        spacings[1:]
    )


# Print statistics for each symmetry sector
def print_sector_statistics(sectors):
    print("\nSymmetry-Sector Statistics")
    print("-" * 58)

    print(
        f"{'Sector':>10}"
        f"{'States':>10}"
        f"{'<r>':>14}"
        f"{'E min':>12}"
        f"{'E max':>12}"
    )

    for sector in ["++", "+-", "-+", "--"]:
        energies = np.asarray(sectors[sector])
        ratios = adjacent_gap_ratios(energies)

        if len(ratios) > 0:
            mean_ratio = np.mean(ratios)
        else:
            mean_ratio = np.nan

        if len(energies) > 0:
            emin = energies[0]
            emax = energies[-1]
        else:
            emin = np.nan
            emax = np.nan

        print(
            f"{sector:>10}"
            f"{len(energies):10d}"
            f"{mean_ratio:14.6f}"
            f"{emin:12.3f}"
            f"{emax:12.3f}"
        )


# Plot parity correlations for every calculated eigenstate
def plot_parity_correlations(states, px_values, py_values):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].scatter(
        states,
        px_values,
        s=18
    )

    axes[0].axhline(
        1.0,
        linestyle="--",
        alpha=0.5
    )

    axes[0].axhline(
        -1.0,
        linestyle="--",
        alpha=0.5
    )

    axes[0].set_ylabel(r"$P_x$")
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].set_title("Stadium Reflection Parities")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(
        states,
        py_values,
        s=18
    )

    axes[1].axhline(
        1.0,
        linestyle="--",
        alpha=0.5
    )

    axes[1].axhline(
        -1.0,
        linestyle="--",
        alpha=0.5
    )

    axes[1].set_xlabel("State")
    axes[1].set_ylabel(r"$P_y$")
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "stadium_parity_correlations.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# Plot adjacent-gap ratios for the four symmetry sectors
def plot_sector_ratios(sectors):
    labels = []
    values = []

    for sector in ["++", "+-", "-+", "--"]:
        energies = np.asarray(sectors[sector])
        ratios = adjacent_gap_ratios(energies)

        labels.append(sector)

        if len(ratios) > 0:
            values.append(np.mean(ratios))
        else:
            values.append(np.nan)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.bar(
        labels,
        values
    )

    # Reference values for orientation
    ax.axhline(
        0.386,
        linestyle="--",
        label="Poisson"
    )

    ax.axhline(
        0.536,
        linestyle=":",
        label="GOE"
    )

    ax.set_xlabel("Reflection symmetry sector")
    ax.set_ylabel(r"Mean adjacent-gap ratio $\langle r \rangle$")
    ax.set_ylim(0.0, 0.65)
    ax.set_title("Stadium Symmetry-Resolved Gap Ratios")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "stadium_symmetry_gap_ratios.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


if __name__ == "__main__":
    number_of_states = 150

    # Equal-area stadium parameters
    stadium_a = 1.0 / np.sqrt(4.0 + np.pi)

    node, elm, boundary, interior, energies, eigenvectors = solve_stadium(
        h=0.03,
        a=stadium_a,
        R=stadium_a,
        n_arc=60
    )

    energies = energies[:number_of_states]
    eigenvectors = eigenvectors[:, :number_of_states]

    print("Stadium Symmetry Analysis")
    print("=" * 52)
    print(f"Nodes             : {len(node)}")
    print(f"Elements          : {len(elm)}")
    print(f"States analyzed   : {len(energies)}")

    # Construct numerical reflection maps
    x_map, x_distance = reflection_map(
        node,
        "x"
    )

    y_map, y_distance = reflection_map(
        node,
        "y"
    )

    print("\nReflection Mapping")
    print("-" * 40)
    print(f"Maximum x-map error : {np.max(x_distance):.8f}")
    print(f"Mean x-map error    : {np.mean(x_distance):.8f}")
    print(f"Maximum y-map error : {np.max(y_distance):.8f}")
    print(f"Mean y-map error    : {np.mean(y_distance):.8f}")

    sectors = {
        "++": [],
        "+-": [],
        "-+": [],
        "--": []
    }

    state_numbers = []
    px_values = []
    py_values = []
    sector_labels = []

    print("\nState Classification")
    print("-" * 62)

    print(
        f"{'State':>8}"
        f"{'Energy':>14}"
        f"{'Px':>12}"
        f"{'Py':>12}"
        f"{'Sector':>10}"
    )

    for i, energy in enumerate(energies):
        psi = full_eigenfunction(
            eigenvectors[:, i],
            interior,
            len(node)
        )

        px = parity_correlation(
            psi,
            x_map
        )

        py = parity_correlation(
            psi,
            y_map
        )

        sector = classify_sector(
            px,
            py,
            threshold=0.8
        )

        state_numbers.append(i + 1)
        px_values.append(px)
        py_values.append(py)
        sector_labels.append(sector)

        if sector in sectors:
            sectors[sector].append(energy)

        print(
            f"{i + 1:8d}"
            f"{energy:14.6f}"
            f"{px:12.6f}"
            f"{py:12.6f}"
            f"{sector:>10}"
        )

    state_numbers = np.asarray(state_numbers)
    px_values = np.asarray(px_values)
    py_values = np.asarray(py_values)

    mixed_count = sector_labels.count("mixed")

    print(f"\nMixed/unclassified states : {mixed_count}")

    print_sector_statistics(
        sectors
    )

    # Save state-by-state classification
    output = DATA_DIR / "stadium_symmetry_classification.csv"

    with open(output, "w") as file:
        file.write(
            "State,Energy,Px,Py,Sector\n"
        )

        for state, energy, px, py, sector in zip(
            state_numbers,
            energies,
            px_values,
            py_values,
            sector_labels
        ):
            file.write(
                f"{state},"
                f"{energy:.10f},"
                f"{px:.10f},"
                f"{py:.10f},"
                f"{sector}\n"
            )

    plot_parity_correlations(
        state_numbers,
        px_values,
        py_values
    )

    plot_sector_ratios(
        sectors
    )

    print("\nSaved classification to:")
    print(output)

    print("\nSaved figures to:")
    print(FIGURES_DIR)