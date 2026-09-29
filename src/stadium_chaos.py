# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:39:42 2026

@author: natoo
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stadium_sector_solver import solve_stadium_sector


# Set output directories
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "results" / "data"
FIGURES_DIR = ROOT / "results" / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Theoretical mean adjacent-gap ratios
R_POISSON = 2.0 * np.log(2.0) - 1.0
R_GOE = 0.5307
R_GOE_SURMISE = 0.53590


# Calculate nearest-neighbor spacings
def spacings(energies):
    return np.diff(
        np.asarray(
            energies,
            dtype=float
        )
    )


# Calculate adjacent-gap ratios without spectral unfolding
def gap_ratios(energies):
    s = spacings(
        energies
    )

    s1 = s[:-1]
    s2 = s[1:]

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


# Bootstrap uncertainty for a collection of gap ratios
def bootstrap_mean_ratio(
    ratios,
    number_of_samples=10000,
    seed=12345
):
    ratios = np.asarray(
        ratios,
        dtype=float
    )

    if len(ratios) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(
        seed
    )

    bootstrap_means = np.empty(
        number_of_samples
    )

    for i in range(number_of_samples):
        sample = rng.choice(
            ratios,
            size=len(ratios),
            replace=True
        )

        bootstrap_means[i] = np.mean(
            sample
        )

    standard_error = np.std(
        bootstrap_means,
        ddof=1
    )

    lower, upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5]
    )

    return (
        standard_error,
        lower,
        upper
    )


# Save one symmetry-sector spectrum
def save_sector_spectrum(sector, energies):
    states = np.arange(
        1,
        len(energies) + 1
    )

    data = np.column_stack([
        states,
        energies
    ])

    np.savetxt(
        DATA_DIR / f"stadium_sector_{sector}.csv",
        data,
        delimiter=",",
        header="State,Energy",
        comments="",
        fmt=["%d", "%.10f"]
    )


# Print statistics over several spectral windows
def print_window_statistics(sector, energies):
    windows = [
        (0, 20),
        (0, 30),
        (0, 40),
        (0, 50),
        (10, 30),
        (20, 50)
    ]

    print(
        f"\nSector {sector}"
    )

    print("-" * 48)

    print(
        f"{'Levels':>14}"
        f"{'States':>10}"
        f"{'<r>':>14}"
    )

    for start, stop in windows:
        stop = min(
            stop,
            len(energies)
        )

        if stop - start < 3:
            continue

        selected = energies[
            start:stop
        ]

        ratio = mean_gap_ratio(
            selected
        )

        print(
            f"{start + 1:4d}-{stop:<7d}"
            f"{len(selected):10d}"
            f"{ratio:14.6f}"
        )


# Plot symmetry-resolved mean gap ratios
def plot_mean_ratios(sector_spectra):
    sectors = list(
        sector_spectra.keys()
    )

    values = [
        mean_gap_ratio(
            sector_spectra[sector]
        )
        for sector in sectors
    ]

    fig, ax = plt.subplots(
        figsize=(7.5, 4.8)
    )

    ax.bar(
        sectors,
        values
    )

    ax.axhline(
        R_POISSON,
        linestyle="--",
        label=f"Poisson ({R_POISSON:.3f})"
    )

    ax.axhline(
        R_GOE,
        linestyle=":",
        label=f"Asymptotic GOE ({R_GOE:.3f})"
    )

    ax.axhline(
        R_GOE_SURMISE,
        linestyle="-.",
        label=f"GOE surmise ({R_GOE_SURMISE:.3f})"
    )

    ax.set_xlabel(
        "Reflection symmetry sector"
    )

    ax.set_ylabel(
        r"Mean adjacent-gap ratio $\langle r \rangle$"
    )

    ax.set_ylim(
        0.30,
        0.64
    )

    ax.set_title(
        "Stadium Symmetry-Resolved Gap Ratios"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "stadium_chaos_gap_ratios.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# Plot the distribution of adjacent-gap ratios
def plot_ratio_distribution(sector_spectra):
    all_ratios = np.concatenate([
        gap_ratios(
            energies
        )
        for energies in sector_spectra.values()
    ])

    r = np.linspace(
        0.0,
        1.0,
        500
    )

    # Poisson ratio distribution
    poisson = (
        2.0
        / (1.0 + r)**2
    )

    # Wigner-like GOE ratio surmise
    goe_surmise = (
        (27.0 / 4.0)
        * (r + r**2)
        / (1.0 + r + r**2)**2.5
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 5.0)
    )

    ax.hist(
        all_ratios,
        bins=20,
        range=(0.0, 1.0),
        density=True,
        alpha=0.65,
        label="Stadium FEM"
    )

    ax.plot(
        r,
        poisson,
        linestyle="--",
        label="Poisson"
    )

    ax.plot(
        r,
        goe_surmise,
        linestyle="-",
        label="GOE ratio surmise"
    )

    ax.set_xlabel(
        r"Adjacent-gap ratio $r$"
    )

    ax.set_ylabel(
        r"$P(r)$"
    )

    ax.set_xlim(
        0.0,
        1.0
    )

    ax.set_title(
        "Stadium Adjacent-Gap Ratio Distribution"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "stadium_gap_ratio_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# Plot stability of the mean gap ratio with level count
def plot_ratio_convergence(sector_spectra):
    counts = np.arange(
        10,
        51,
        5
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for sector, energies in sector_spectra.items():
        values = []

        for count in counts:
            n = min(
                count,
                len(energies)
            )

            values.append(
                mean_gap_ratio(
                    energies[:n]
                )
            )

        ax.plot(
            counts,
            values,
            marker="o",
            markersize=3,
            label=sector
        )

    ax.axhline(
        R_POISSON,
        linestyle="--",
        label="Poisson"
    )

    ax.axhline(
        R_GOE,
        linestyle=":",
        label="Asymptotic GOE"
    )

    ax.axhline(
        R_GOE_SURMISE,
        linestyle="-.",
        label="GOE surmise"
    )

    ax.set_xlabel(
        "Number of levels included"
    )

    ax.set_ylabel(
        r"Mean adjacent-gap ratio $\langle r \rangle$"
    )

    ax.set_title(
        "Stadium Gap-Ratio Stability"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "stadium_gap_ratio_convergence.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


if __name__ == "__main__":
    # Use only the low-energy portion of a fine mesh
    number_of_states = 50
    mesh_size = 0.01

    a = 1.0 / np.sqrt(
        4.0 + np.pi
    )

    sectors = [
        "++",
        "+-",
        "-+",
        "--"
    ]

    sector_spectra = {}

    print("Stadium Quantum-Chaos Analysis")
    print("=" * 60)

    print(
        f"Mesh h             : {mesh_size:.3f}"
    )

    print(
        f"States per sector   : {number_of_states}"
    )

    print(
        f"Poisson <r>         : {R_POISSON:.6f}"
    )

    print(
        f"Asymptotic GOE      : {R_GOE:.6f}"
    )

    print(
        f"GOE surmise         : {R_GOE_SURMISE:.6f}"
    )

    # Solve each reflection-symmetry sector independently
    for sector in sectors:
        print(
            f"\nSolving sector {sector}..."
        )

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
            h=mesh_size,
            a=a,
            R=a,
            n_arc=80,
            num_eigenvalues=number_of_states
        )

        sector_spectra[
            sector
        ] = energies

        save_sector_spectrum(
            sector,
            energies
        )

        print(
            f"DOF             : {len(free_nodes)}"
        )

        print(
            f"States retained : {len(energies)}"
        )

        print(
            f"E1              : {energies[0]:.6f}"
        )

        print(
            f"E{len(energies)}             : "
            f"{energies[-1]:.6f}"
        )

        print(
            f"<r>             : "
            f"{mean_gap_ratio(energies):.6f}"
        )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "SPECTRAL-WINDOW CHECK"
    )

    print(
        "=" * 60
    )

    for sector in sectors:
        print_window_statistics(
            sector,
            sector_spectra[sector]
        )

    # Combine ratios only after calculating them within each sector
    combined_ratios = np.concatenate([
        gap_ratios(
            sector_spectra[sector]
        )
        for sector in sectors
    ])

    combined_mean = np.mean(
        combined_ratios
    )

    # Estimate statistical uncertainty by bootstrap resampling
    (
        bootstrap_se,
        confidence_lower,
        confidence_upper
    ) = bootstrap_mean_ratio(
        combined_ratios,
        number_of_samples=10000,
        seed=12345
    )

    print(
        "\nCombined within-sector statistics"
    )

    print(
        "-" * 48
    )

    print(
        f"Gap ratios       : {len(combined_ratios)}"
    )

    print(
        f"Mean <r>         : {combined_mean:.6f}"
    )

    print(
        f"Bootstrap SE     : {bootstrap_se:.6f}"
    )

    print(
        f"95% bootstrap CI : "
        f"[{confidence_lower:.6f}, "
        f"{confidence_upper:.6f}]"
    )

    print(
        f"Poisson          : {R_POISSON:.6f}"
    )

    print(
        f"Asymptotic GOE   : {R_GOE:.6f}"
    )

    print(
        f"GOE surmise      : {R_GOE_SURMISE:.6f}"
    )

    plot_mean_ratios(
        sector_spectra
    )

    plot_ratio_distribution(
        sector_spectra
    )

    plot_ratio_convergence(
        sector_spectra
    )

    print(
        "\nSaved spectra to:"
    )

    print(
        DATA_DIR
    )

    print(
        "\nSaved figures to:"
    )

    print(
        FIGURES_DIR
    )