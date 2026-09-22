# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:39:31 2026

@author: natoo
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from square_solver import solve_square, exact_square_energy
from circle_solver import solve_circle, exact_circle_energy
from hexagon_solver import solve_hexagon
from stadium_solver import solve_stadium


# Output locations
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "results" / "data"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# Calculate the energy change between consecutive mesh refinements
def add_refinement_change(results):
    results[0]["change_percent"] = np.nan

    for i in range(1, len(results)):
        previous = results[i - 1]["energy"]
        current = results[i]["energy"]
        results[i]["change_percent"] = abs(current - previous) / current * 100.0

    return results


# Run the square convergence study using the analytical ground state
def square_convergence():
    mesh_sizes = [5, 10, 20, 30, 40, 50]
    exact_energy = exact_square_energy(1, 1)
    results = []

    print("\nSquare Convergence")
    print("-" * 68)
    print(
        f"{'N':>6}{'Nodes':>10}{'E1':>14}{'Exact':>14}"
        f"{'Error (%)':>12}{'Change (%)':>12}"
    )

    for N in mesh_sizes:
        node, elm, boundary, interior, energies, eigenvectors = solve_square(N)

        energy = energies[0]
        error = abs(energy - exact_energy) / exact_energy * 100.0

        results.append({
            "resolution": N,
            "nodes": len(node),
            "energy": energy,
            "exact": exact_energy,
            "error": error
        })

    add_refinement_change(results)

    for result in results:
        change = result["change_percent"]
        change_text = "-" if np.isnan(change) else f"{change:.5f}"

        print(
            f"{result['resolution']:6d}"
            f"{result['nodes']:10d}"
            f"{result['energy']:14.8f}"
            f"{result['exact']:14.8f}"
            f"{result['error']:12.5f}"
            f"{change_text:>12}"
        )

    return results


# Run the circle convergence study using the analytical Bessel solution
def circle_convergence():
    spacings = [0.10, 0.08, 0.06, 0.05, 0.04, 0.03]
    radius = 1.0 / np.sqrt(np.pi)
    exact_energy = exact_circle_energy(0, 1, radius)
    results = []

    print("\nCircle Convergence")
    print("-" * 72)
    print(
        f"{'Spacing':>10}{'Nodes':>10}{'E1':>14}{'Exact':>14}"
        f"{'Error (%)':>12}{'Change (%)':>12}"
    )

    for spacing in spacings:
        node, elm, boundary, interior, energies, eigenvectors = solve_circle(
            spacing=spacing,
            radius=radius,
            n_boundary=100
        )

        energy = energies[0]
        error = abs(energy - exact_energy) / exact_energy * 100.0

        results.append({
            "resolution": spacing,
            "nodes": len(node),
            "energy": energy,
            "exact": exact_energy,
            "error": error
        })

    add_refinement_change(results)

    for result in results:
        change = result["change_percent"]
        change_text = "-" if np.isnan(change) else f"{change:.5f}"

        print(
            f"{result['resolution']:10.3f}"
            f"{result['nodes']:10d}"
            f"{result['energy']:14.8f}"
            f"{result['exact']:14.8f}"
            f"{result['error']:12.5f}"
            f"{change_text:>12}"
        )

    return results


# Track numerical convergence for the equal-area hexagon
def hexagon_convergence():
    spacings = [0.10, 0.08, 0.06, 0.05, 0.04, 0.03]
    radius = np.sqrt(2.0 / (3.0 * np.sqrt(3.0)))
    results = []

    print("\nHexagon Convergence")
    print("-" * 52)
    print(
        f"{'Spacing':>10}{'Nodes':>10}{'E1':>18}"
        f"{'Change (%)':>14}"
    )

    for spacing in spacings:
        node, elm, boundary, interior, energies, eigenvectors = solve_hexagon(
            spacing=spacing,
            radius=radius
        )

        results.append({
            "resolution": spacing,
            "nodes": len(node),
            "energy": energies[0]
        })

    add_refinement_change(results)

    for result in results:
        change = result["change_percent"]
        change_text = "-" if np.isnan(change) else f"{change:.5f}"

        print(
            f"{result['resolution']:10.3f}"
            f"{result['nodes']:10d}"
            f"{result['energy']:18.8f}"
            f"{change_text:>14}"
        )

    return results


# Track numerical convergence for the equal-area stadium
def stadium_convergence():
    spacings = [0.10, 0.08, 0.06, 0.05, 0.04, 0.03]
    a = 1.0 / np.sqrt(4.0 + np.pi)
    R = a
    results = []

    print("\nStadium Convergence")
    print("-" * 52)
    print(
        f"{'Spacing':>10}{'Nodes':>10}{'E1':>18}"
        f"{'Change (%)':>14}"
    )

    for spacing in spacings:
        node, elm, boundary, interior, energies, eigenvectors = solve_stadium(
            h=spacing,
            a=a,
            R=R,
            n_arc=60
        )

        results.append({
            "resolution": spacing,
            "nodes": len(node),
            "energy": energies[0]
        })

    add_refinement_change(results)

    for result in results:
        change = result["change_percent"]
        change_text = "-" if np.isnan(change) else f"{change:.5f}"

        print(
            f"{result['resolution']:10.3f}"
            f"{result['nodes']:10d}"
            f"{result['energy']:18.8f}"
            f"{change_text:>14}"
        )

    return results


# Save each convergence study as a CSV file
def save_results(name, results):
    output_file = DATA_DIR / f"{name}_convergence.csv"
    fieldnames = results[0].keys()

    with output_file.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {output_file}")


# Plot analytical error for the square and circle
def plot_validation(square_results, circle_results):
    square_nodes = [result["nodes"] for result in square_results]
    square_errors = [result["error"] for result in square_results]

    circle_nodes = [result["nodes"] for result in circle_results]
    circle_errors = [result["error"] for result in circle_results]

    plt.figure(figsize=(7, 5))

    plt.loglog(square_nodes, square_errors, "o-", label="Square")
    plt.loglog(circle_nodes, circle_errors, "s-", label="Circle")

    plt.xlabel("Number of mesh nodes")
    plt.ylabel("Ground-state relative error (%)")
    plt.title("FEM Validation Against Analytical Solutions")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_file = FIGURE_DIR / "analytical_validation.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved {output_file}")


# Plot ground-state convergence for the spacing-based geometries
def plot_ground_state_convergence(circle_results, hexagon_results, stadium_results):
    plt.figure(figsize=(7, 5))

    for name, results in [
        ("Circle", circle_results),
        ("Hexagon", hexagon_results),
        ("Stadium", stadium_results)
    ]:
        spacing = [result["resolution"] for result in results]
        energy = [result["energy"] for result in results]

        plt.plot(spacing, energy, "o-", label=name)

    plt.xlabel("Mesh spacing")
    plt.ylabel("Ground-state energy")
    plt.title("Ground-State Mesh Convergence")
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_file = FIGURE_DIR / "ground_state_convergence.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved {output_file}")


def main():
    square_results = square_convergence()
    circle_results = circle_convergence()
    hexagon_results = hexagon_convergence()
    stadium_results = stadium_convergence()

    save_results("square", square_results)
    save_results("circle", circle_results)
    save_results("hexagon", hexagon_results)
    save_results("stadium", stadium_results)

    plot_validation(square_results, circle_results)
    plot_ground_state_convergence(
        circle_results,
        hexagon_results,
        stadium_results
    )


if __name__ == "__main__":
    main()