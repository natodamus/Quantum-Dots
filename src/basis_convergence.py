# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:18:48 2026

@author: natoo
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hexagon_basis import solve_hexagon_basis
from stadium_basis import solve_stadium_basis
from hexagon_solver import solve_hexagon
from stadium_solver import solve_stadium


# Set output directories
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "results" / "data"
FIGURES_DIR = ROOT / "results" / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Run a basis-size convergence study
def basis_size_study(solver, basis_sizes, grid_size=260, V0=1e4):
    results = []

    for M in basis_sizes:
        energies = solver(
            M=M,
            grid_size=grid_size,
            V0=V0,
            num_states=10
        )[0]

        results.append([
            M,
            M * M,
            energies[0],
            energies[1],
            energies[2],
            energies[4],
            energies[9]
        ])

    return np.asarray(results)


# Run a barrier-height study at fixed basis size
def barrier_study(solver, barrier_values, M=18, grid_size=220):
    results = []

    for V0 in barrier_values:
        energies = solver(
            M=M,
            grid_size=grid_size,
            V0=V0,
            num_states=5
        )[0]

        results.append([
            V0,
            energies[0],
            energies[1],
            energies[2],
            energies[4]
        ])

    return np.asarray(results)


# Save a numerical table to CSV
def save_csv(filename, data, header):
    np.savetxt(
        DATA_DIR / filename,
        data,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.10f"
    )


# Print the basis-size results
def print_basis_table(name, data):
    print(f"\n{name} Basis-Size Study")
    print("-" * 80)

    print(
        f"{'M':>6}"
        f"{'Basis':>10}"
        f"{'E1':>14}"
        f"{'E2':>14}"
        f"{'E3':>14}"
        f"{'E5':>14}"
        f"{'E10':>14}"
    )

    for row in data:
        print(
            f"{int(row[0]):6d}"
            f"{int(row[1]):10d}"
            f"{row[2]:14.6f}"
            f"{row[3]:14.6f}"
            f"{row[4]:14.6f}"
            f"{row[5]:14.6f}"
            f"{row[6]:14.6f}"
        )


# Print the barrier-height results
def print_barrier_table(name, data):
    print(f"\n{name} Barrier Study")
    print("-" * 74)

    print(
        f"{'V0':>12}"
        f"{'E1':>14}"
        f"{'E2':>14}"
        f"{'E3':>14}"
        f"{'E5':>14}"
    )

    for row in data:
        print(
            f"{row[0]:12.0f}"
            f"{row[1]:14.6f}"
            f"{row[2]:14.6f}"
            f"{row[3]:14.6f}"
            f"{row[4]:14.6f}"
        )


# Plot ground-state energy versus basis size
def plot_basis_convergence(hexagon, stadium, hex_fem, stadium_fem):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(
        hexagon[:, 1],
        hexagon[:, 2],
        marker="o",
        label="Basis"
    )

    axes[0].axhline(
        hex_fem,
        linestyle="--",
        label="FEM"
    )

    axes[0].set_title("Hexagon")
    axes[0].set_xlabel("Number of basis functions")
    axes[0].set_ylabel("Ground-state energy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        stadium[:, 1],
        stadium[:, 2],
        marker="o",
        label="Basis"
    )

    axes[1].axhline(
        stadium_fem,
        linestyle="--",
        label="FEM"
    )

    axes[1].set_title("Stadium")
    axes[1].set_xlabel("Number of basis functions")
    axes[1].set_ylabel("Ground-state energy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        r"Basis-Size Study at $V_0 = 10^4$"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "basis_size_convergence.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# Plot ground-state energy versus barrier height
def plot_barrier_study(hexagon, stadium, hex_fem, stadium_fem):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].semilogx(
        hexagon[:, 0],
        hexagon[:, 1],
        marker="o",
        label="Basis"
    )

    axes[0].axhline(
        hex_fem,
        linestyle="--",
        label="FEM"
    )

    axes[0].set_title("Hexagon")
    axes[0].set_xlabel(r"Barrier height $V_0$")
    axes[0].set_ylabel("Ground-state energy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].semilogx(
        stadium[:, 0],
        stadium[:, 1],
        marker="o",
        label="Basis"
    )

    axes[1].axhline(
        stadium_fem,
        linestyle="--",
        label="FEM"
    )

    axes[1].set_title("Stadium")
    axes[1].set_xlabel(r"Barrier height $V_0$")
    axes[1].set_ylabel("Ground-state energy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        r"Barrier Study with an $18 \times 18$ Sine Basis"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "basis_barrier_study.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


if __name__ == "__main__":
    basis_sizes = [10, 12, 15, 18, 20, 22, 25]
    barrier_values = [1e3, 1e4, 1e5, 1e6]

    # Calculate final FEM reference values
    hexagon_radius = np.sqrt(2.0 / (3.0 * np.sqrt(3.0)))

    hexagon_fem = solve_hexagon(
        spacing=0.03,
        radius=hexagon_radius
    )[4][0]

    stadium_a = 1.0 / np.sqrt(4.0 + np.pi)

    stadium_fem = solve_stadium(
        h=0.03,
        a=stadium_a,
        R=stadium_a,
        n_arc=60
    )[4][0]

    print("FEM Reference Energies")
    print("-" * 30)
    print(f"Hexagon E1 : {hexagon_fem:.8f}")
    print(f"Stadium E1 : {stadium_fem:.8f}")

    # Run basis-size studies at fixed barrier height
    hexagon_basis = basis_size_study(
        solve_hexagon_basis,
        basis_sizes,
        grid_size=260,
        V0=1e4
    )

    stadium_basis = basis_size_study(
        solve_stadium_basis,
        basis_sizes,
        grid_size=260,
        V0=1e4
    )

    # Run barrier studies at fixed basis size
    hexagon_barrier = barrier_study(
        solve_hexagon_basis,
        barrier_values,
        M=18,
        grid_size=220
    )

    stadium_barrier = barrier_study(
        solve_stadium_basis,
        barrier_values,
        M=18,
        grid_size=220
    )

    print_basis_table(
        "Hexagon",
        hexagon_basis
    )

    print_basis_table(
        "Stadium",
        stadium_basis
    )

    print_barrier_table(
        "Hexagon",
        hexagon_barrier
    )

    print_barrier_table(
        "Stadium",
        stadium_barrier
    )

    # Save all numerical results
    save_csv(
        "hexagon_basis_convergence.csv",
        hexagon_basis,
        "M,Basis Functions,E1,E2,E3,E5,E10"
    )

    save_csv(
        "stadium_basis_convergence.csv",
        stadium_basis,
        "M,Basis Functions,E1,E2,E3,E5,E10"
    )

    save_csv(
        "hexagon_barrier_study.csv",
        hexagon_barrier,
        "V0,E1,E2,E3,E5"
    )

    save_csv(
        "stadium_barrier_study.csv",
        stadium_barrier,
        "V0,E1,E2,E3,E5"
    )

    # Generate final basis-method figures
    plot_basis_convergence(
        hexagon_basis,
        stadium_basis,
        hexagon_fem,
        stadium_fem
    )

    plot_barrier_study(
        hexagon_barrier,
        stadium_barrier,
        hexagon_fem,
        stadium_fem
    )

    print("\nSaved basis-method results to:")
    print(DATA_DIR)

    print("\nSaved basis-method figures to:")
    print(FIGURES_DIR)