# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:30:02 2026

@author: natoo
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from square_solver import solve_square
from circle_solver import solve_circle
from hexagon_solver import solve_hexagon
from stadium_solver import solve_stadium


# Set output directories
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "results" / "data"
FIGURES_DIR = ROOT / "results" / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Calculate nearest-neighbor energy spacings
def nearest_neighbor_spacings(energies):
    return np.diff(energies)


# Normalize spacings by their mean value
def normalized_spacings(energies):
    spacings = nearest_neighbor_spacings(energies)
    return spacings / np.mean(spacings)


# Calculate the adjacent-gap ratio
def adjacent_gap_ratios(energies):
    spacings = nearest_neighbor_spacings(energies)

    return np.minimum(
        spacings[:-1],
        spacings[1:]
    ) / np.maximum(
        spacings[:-1],
        spacings[1:]
    )


# Save an energy spectrum to CSV
def save_spectrum(name, energies):
    states = np.arange(1, len(energies) + 1)

    data = np.column_stack([
        states,
        energies
    ])

    np.savetxt(
        DATA_DIR / f"{name}_spectrum.csv",
        data,
        delimiter=",",
        header="State,Energy",
        comments="",
        fmt=["%d", "%.10f"]
    )


# Print basic spectral statistics
def print_statistics(name, energies):
    spacings = nearest_neighbor_spacings(energies)
    ratios = adjacent_gap_ratios(energies)

    print(f"\n{name}")
    print("-" * 40)
    print(f"States              : {len(energies)}")
    print(f"Lowest energy       : {energies[0]:.6f}")
    print(f"Highest energy      : {energies[-1]:.6f}")
    print(f"Mean raw spacing    : {np.mean(spacings):.6f}")
    print(f"Mean adjacent ratio : {np.mean(ratios):.6f}")
    print(f"Minimum spacing     : {np.min(spacings):.6f}")


# Plot the first part of each spectrum
def plot_spectra(spectra, number_to_plot=50):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, (name, energies) in zip(
        axes.ravel(),
        spectra.items()
    ):
        n = min(number_to_plot, len(energies))
        states = np.arange(1, n + 1)

        ax.plot(
            states,
            energies[:n],
            marker="o",
            markersize=3
        )

        ax.set_title(name)
        ax.set_xlabel("State")
        ax.set_ylabel("Energy")
        ax.grid(alpha=0.3)

    fig.suptitle("Low-Energy Quantum-Dot Spectra")
    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "spectral_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# Plot raw normalized nearest-neighbor spacing histograms
def plot_spacing_histograms(spectra):
    x = np.linspace(0.0, 3.0, 400)

    # Reference distributions shown only for orientation
    poisson = np.exp(-x)
    goe = 0.5 * np.pi * x * np.exp(-0.25 * np.pi * x**2)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, (name, energies) in zip(
        axes.ravel(),
        spectra.items()
    ):
        spacings = normalized_spacings(energies)

        ax.hist(
            spacings,
            bins=20,
            density=True,
            alpha=0.6,
            label="FEM"
        )

        ax.plot(
            x,
            poisson,
            linestyle="--",
            label="Poisson"
        )

        ax.plot(
            x,
            goe,
            linestyle="-",
            label="GOE"
        )

        ax.set_title(name)
        ax.set_xlabel(r"$s / \langle s \rangle$")
        ax.set_ylabel(r"$P(s)$")
        ax.set_xlim(0.0, 3.0)
        ax.legend()

    fig.suptitle(
        "Raw Nearest-Neighbor Spacing Distributions"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "raw_spacing_distributions.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


if __name__ == "__main__":
    number_of_states = 150

    print("Calculating FEM spectra...")
    print("This may take a little while.")

    # Square
    square_energies = solve_square(
        N=50
    )[4][:number_of_states]

    print("Square complete.")

    # Equal-area circle
    circle_radius = 1.0 / np.sqrt(np.pi)

    circle_energies = solve_circle(
        spacing=0.03,
        radius=circle_radius,
        n_boundary=100
    )[4][:number_of_states]

    print("Circle complete.")

    # Equal-area regular hexagon
    hexagon_radius = np.sqrt(
        2.0 / (3.0 * np.sqrt(3.0))
    )

    hexagon_energies = solve_hexagon(
        spacing=0.03,
        radius=hexagon_radius
    )[4][:number_of_states]

    print("Hexagon complete.")

    # Equal-area stadium
    stadium_a = 1.0 / np.sqrt(4.0 + np.pi)

    stadium_energies = solve_stadium(
        h=0.03,
        a=stadium_a,
        R=stadium_a,
        n_arc=60
    )[4][:number_of_states]

    print("Stadium complete.")

    spectra = {
        "Square": square_energies,
        "Circle": circle_energies,
        "Hexagon": hexagon_energies,
        "Stadium": stadium_energies
    }

    print("\n" + "=" * 52)
    print("RAW SPECTRAL STATISTICS")
    print("=" * 52)

    for name, energies in spectra.items():
        print_statistics(
            name,
            energies
        )

    # Save spectra for later symmetry and unfolding analysis
    for name, energies in spectra.items():
        save_spectrum(
            name.lower(),
            energies
        )

    plot_spectra(spectra)

    plot_spacing_histograms(spectra)

    print("\nSaved spectra to:")
    print(DATA_DIR)

    print("\nSaved figures to:")
    print(FIGURES_DIR)

    print("\nImportant:")
    print(
        "These spacing distributions mix symmetry sectors and use only "
        "mean-spacing normalization."
    )
    print(
        "They are exploratory and should not yet be interpreted as "
        "evidence for or against quantum chaos."
    )