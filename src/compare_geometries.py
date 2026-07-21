# -*- coding: utf-8 -*-
"""
Compare the energy spectra of square, circular, and hexagonal
two-dimensional quantum dots having the same area.

Created on Tue Jul 14 13:06:25 2026

@author: Renato R. Silva
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from FEM import A_mat, B_mat
from circle_mesh import circle_mesh
from hexagon_mesh import hexagon_mesh


# =========================================================
# Project directories
# =========================================================

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "results" / "figures"
DATA_DIR = ROOT / "results" / "data"


def create_output_directories() -> None:
    """Create the project output directories when necessary."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Square mesh generator
# =========================================================

def square_mesh(
    subdivisions: int,
    side_length: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a triangular mesh for a square domain.

    Parameters
    ----------
    subdivisions
        Number of subdivisions along each side.

    side_length
        Length of each side of the square.

    Returns
    -------
    node
        Array containing the node coordinates.

    elm
        Array containing the triangular element connectivity.

    boundary
        Indices of nodes located on the square boundary.
    """

    node = []

    for j in range(subdivisions + 1):
        for i in range(subdivisions + 1):
            x = side_length * i / subdivisions
            y = side_length * j / subdivisions
            node.append([x, y])

    node = np.asarray(node, dtype=float)

    elm = []

    for j in range(subdivisions):
        for i in range(subdivisions):
            n0 = j * (subdivisions + 1) + i
            n1 = n0 + 1
            n2 = n0 + subdivisions + 1
            n3 = n2 + 1

            elm.append([n0, n1, n3])
            elm.append([n0, n3, n2])

    elm = np.asarray(elm, dtype=int)

    tolerance = 1.0e-12

    boundary = np.where(
        np.isclose(node[:, 0], 0.0, atol=tolerance)
        | np.isclose(node[:, 0], side_length, atol=tolerance)
        | np.isclose(node[:, 1], 0.0, atol=tolerance)
        | np.isclose(node[:, 1], side_length, atol=tolerance)
    )[0]

    return node, elm, boundary


# =========================================================
# FEM eigenvalue solver
# =========================================================

def solve_geometry(
    node: np.ndarray,
    elm: np.ndarray,
    boundary: np.ndarray,
    number_of_states: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the infinite-well Schrödinger equation for a mesh.

    The finite-element generalized eigenvalue problem is

        (1/2) A psi = E B psi.

    Dirichlet boundary conditions are imposed by restricting the
    stiffness and mass matrices to the interior nodes.

    Parameters
    ----------
    node
        Node coordinates.

    elm
        Triangular element connectivity.

    boundary
        Indices of boundary nodes.

    number_of_states
        Number of low-energy eigenstates to calculate.

    Returns
    -------
    energies
        Calculated energy eigenvalues.

    eigenvectors
        Interior-node eigenvectors.

    interior
        Indices of interior nodes.
    """

    stiffness_matrix = A_mat(node, elm)
    mass_matrix = B_mat(node, elm)

    all_nodes = np.arange(len(node))
    interior = np.setdiff1d(all_nodes, boundary)

    stiffness_interior = stiffness_matrix[np.ix_(interior, interior)]
    mass_interior = mass_matrix[np.ix_(interior, interior)]

    energies, eigenvectors = eigh(
        0.5 * stiffness_interior,
        mass_interior,
        subset_by_index=[0, number_of_states - 1]
    )

    return energies, eigenvectors, interior


# =========================================================
# Mesh calculations
# =========================================================

def calculate_mesh_area(
    node: np.ndarray,
    elm: np.ndarray
) -> float:
    """
    Calculate the total area represented by a triangular mesh.
    """

    total_area = 0.0

    for triangle in elm:
        point_1, point_2, point_3 = node[triangle]

        triangle_area = 0.5 * abs(
            (point_2[0] - point_1[0])
            * (point_3[1] - point_1[1])
            - (point_3[0] - point_1[0])
            * (point_2[1] - point_1[1])
        )

        total_area += triangle_area

    return total_area


def calculate_equal_area_dimensions(
    target_area: float
) -> dict[str, float]:
    """
    Calculate the dimensions required for equal-area geometries.
    """

    return {
        "square_side": np.sqrt(target_area),
        "circle_radius": np.sqrt(target_area / np.pi),
        "hexagon_radius": np.sqrt(
            2.0 * target_area / (3.0 * np.sqrt(3.0))
        )
    }


def generate_equal_area_meshes(
    target_area: float,
    mesh_spacing: float
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate square, circular, and hexagonal meshes with equal areas.
    """

    dimensions = calculate_equal_area_dimensions(target_area)

    square_side = dimensions["square_side"]
    circle_radius = dimensions["circle_radius"]
    hexagon_radius = dimensions["hexagon_radius"]

    square_subdivisions = round(square_side / mesh_spacing)

    square_geometry = square_mesh(
        subdivisions=square_subdivisions,
        side_length=square_side
    )

    circle_geometry = circle_mesh(
        spacing=mesh_spacing,
        radius=circle_radius,
        center=(0.0, 0.0),
        n_boundary=100
    )

    hexagon_geometry = hexagon_mesh(
        spacing=mesh_spacing,
        radius=hexagon_radius,
        center=(0.0, 0.0)
    )

    return {
        "Square": square_geometry,
        "Circle": circle_geometry,
        "Hexagon": hexagon_geometry
    }


# =========================================================
# Numerical solution
# =========================================================

def solve_all_geometries(
    geometries: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    number_of_states: int
) -> dict[str, np.ndarray]:
    """
    Solve the eigenvalue problem for every supplied geometry.
    """

    energies = {}

    for geometry_name, geometry_data in geometries.items():
        node, elm, boundary = geometry_data

        geometry_energies, _, _ = solve_geometry(
            node=node,
            elm=elm,
            boundary=boundary,
            number_of_states=number_of_states
        )

        energies[geometry_name] = geometry_energies

    return energies


# =========================================================
# Console output
# =========================================================

def print_equal_area_dimensions(target_area: float) -> None:
    """Print the dimensions of the equal-area geometries."""

    dimensions = calculate_equal_area_dimensions(target_area)

    print("Equal-area dimensions")
    print("-" * 45)
    print(f"Target area:              {target_area:.6f}")
    print(
        f"Square side length:       "
        f"{dimensions['square_side']:.6f}"
    )
    print(
        f"Circle radius:            "
        f"{dimensions['circle_radius']:.6f}"
    )
    print(
        f"Hexagon circumradius:     "
        f"{dimensions['hexagon_radius']:.6f}"
    )


def print_mesh_information(
    geometries: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray]
    ]
) -> None:
    """Print node counts, element counts, and numerical mesh areas."""

    print("\nNumerical mesh information")
    print("-" * 70)
    print(
        f"{'Geometry':<12}"
        f"{'Nodes':>10}"
        f"{'Elements':>12}"
        f"{'Mesh area':>14}"
    )

    for geometry_name, geometry_data in geometries.items():
        node, elm, _ = geometry_data
        area = calculate_mesh_area(node, elm)

        print(
            f"{geometry_name:<12}"
            f"{len(node):>10}"
            f"{len(elm):>12}"
            f"{area:>14.6f}"
        )


def print_energy_comparison(
    energies: dict[str, np.ndarray],
    number_of_states: int
) -> None:
    """Print the energy spectra for all geometries."""

    print("\nEqual-area energy comparison")
    print("-" * 62)

    print(
        f"{'State':>7}"
        f"{'Square':>16}"
        f"{'Circle':>16}"
        f"{'Hexagon':>16}"
    )

    for state in range(number_of_states):
        print(
            f"{state + 1:7d}"
            f"{energies['Square'][state]:16.6f}"
            f"{energies['Circle'][state]:16.6f}"
            f"{energies['Hexagon'][state]:16.6f}"
        )


# =========================================================
# Save numerical results
# =========================================================

def save_energy_comparison_csv(
    energies: dict[str, np.ndarray],
    number_of_states: int
) -> Path:
    """
    Save the calculated energies to a CSV file.

    Returns
    -------
    output_path
        Location of the generated CSV file.
    """

    output_path = DATA_DIR / "equal_area_energy_comparison.csv"

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow([
            "State",
            "Square Energy",
            "Circle Energy",
            "Hexagon Energy"
        ])

        for state in range(number_of_states):
            writer.writerow([
                state + 1,
                energies["Square"][state],
                energies["Circle"][state],
                energies["Hexagon"][state]
            ])

    return output_path


# =========================================================
# Plotting
# =========================================================

def plot_energy_spectra(
    energies: dict[str, np.ndarray],
    number_of_states: int
) -> Path:
    """
    Plot and save the energy spectra of the three geometries.
    """

    states = np.arange(1, number_of_states + 1)

    plt.figure(figsize=(8, 6))

    plt.plot(
        states,
        energies["Square"],
        marker="o",
        label="Square"
    )

    plt.plot(
        states,
        energies["Circle"],
        marker="s",
        label="Circle"
    )

    plt.plot(
        states,
        energies["Hexagon"],
        marker="^",
        label="Hexagon"
    )

    plt.xlabel("State number")
    plt.ylabel("Energy")
    plt.title("Energy Spectra of Equal-Area Quantum Dots")
    plt.xticks(states)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = FIGURES_DIR / "equal_area_energy_spectra.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    return output_path


def plot_ground_state_comparison(
    energies: dict[str, np.ndarray]
) -> Path:
    """
    Plot and save the ground-state energies of the geometries.
    """

    geometry_names = ["Square", "Circle", "Hexagon"]

    ground_state_energies = [
        energies["Square"][0],
        energies["Circle"][0],
        energies["Hexagon"][0]
    ]

    plt.figure(figsize=(7, 5))

    plt.bar(
        geometry_names,
        ground_state_energies
    )

    plt.ylabel("Ground-state energy")
    plt.title("Equal-Area Ground-State Energy Comparison")
    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "equal_area_ground_state_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    return output_path


# =========================================================
# Main program
# =========================================================

def main() -> None:
    """Run the complete equal-area geometry comparison."""

    target_area = 1.0
    mesh_spacing = 0.04
    number_of_states = 10

    create_output_directories()

    print_equal_area_dimensions(target_area)

    geometries = generate_equal_area_meshes(
        target_area=target_area,
        mesh_spacing=mesh_spacing
    )

    print_mesh_information(geometries)

    energies = solve_all_geometries(
        geometries=geometries,
        number_of_states=number_of_states
    )

    print_energy_comparison(
        energies=energies,
        number_of_states=number_of_states
    )

    csv_path = save_energy_comparison_csv(
        energies=energies,
        number_of_states=number_of_states
    )

    spectra_path = plot_energy_spectra(
        energies=energies,
        number_of_states=number_of_states
    )

    ground_state_path = plot_ground_state_comparison(
        energies=energies
    )

    print("\nGenerated output files")
    print("-" * 45)
    print(f"Energy data:              {csv_path}")
    print(f"Energy spectra figure:    {spectra_path}")
    print(f"Ground-state figure:      {ground_state_path}")


if __name__ == "__main__":
    main()