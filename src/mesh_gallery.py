# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 21:33:16 2026

@author: natoo

"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from square_solver import square_mesh
from circle_mesh import circle_mesh
from hexagon_mesh import hexagon_mesh
from stadium_mesh import stadium_mesh


# Locate the project root and create the output directory.
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"


def generate_gallery(spacing=0.08, n_arc=80):

    # Create the output directory if it does not exist.
    OUT.mkdir(parents=True, exist_ok=True)

    # Include the mesh spacing in every filename.
    # For example, h=0.08 becomes h080.
    suffix = f"h{round(spacing * 1000):03d}"

    # All four geometries have equal area A = 1.
    square_length = 1.0
    circle_radius = 1 / np.sqrt(np.pi)
    hexagon_radius = np.sqrt(2 / (3 * np.sqrt(3)))
    stadium_radius = 1 / np.sqrt(4 + np.pi)

    # The square mesh uses N subdivisions along each side.
    # Its actual grid spacing is 1/N.
    square_N = max(3, round(1 / spacing))

    # Generate the triangular meshes for each geometry.
    geometries = [
        (
            "square",
            square_mesh(
                square_N,
                length=square_length
            )
        ),
        (
            "circle",
            circle_mesh(
                spacing=spacing,
                radius=circle_radius
            )[:2]
        ),
        (
            "hexagon",
            hexagon_mesh(
                spacing=spacing,
                radius=hexagon_radius
            )[:2]
        ),
        (
            "stadium_full",
            stadium_mesh(
                h=spacing,
                a=stadium_radius,
                R=stadium_radius,
                n_arc=n_arc
            )
        )
    ]

    # Create the combined four-panel mesh figure.
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Store the mesh counts for the final summary.
    results = []

    for ax, (name, (nodes, triangles)) in zip(
        axes.flat, geometries
    ):

        # Count the nodes and triangular elements.
        num_nodes = len(nodes)
        num_triangles = len(triangles)

        results.append(
            (name, num_nodes, num_triangles)
        )

        # Plot the triangular elements in the gallery.
        ax.triplot(
            nodes[:, 0],
            nodes[:, 1],
            triangles,
            lw=0.42 if spacing >= 0.04 else 0.20,
            color="navy"
        )

        ax.set_aspect("equal", adjustable="box")

        ax.set_title(
            f"{name.replace('_', ' ').title()} | "
            f"{num_nodes} nodes, "
            f"{num_triangles} triangles"
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Generate an individual figure for each geometry.
        fig_single, single_ax = plt.subplots(
            figsize=(7, 6)
        )

        single_ax.triplot(
            nodes[:, 0],
            nodes[:, 1],
            triangles,
            lw=0.55 if spacing >= 0.04 else 0.25,
            color="navy"
        )

        single_ax.set_aspect(
            "equal",
            adjustable="box"
        )

        # Report the actual square spacing, since N is rounded.
        if name == "square":
            resolution_label = (
                f"N={square_N}, h={1 / square_N:.4f}"
            )
        else:
            resolution_label = f"h={spacing:g}"

        single_ax.set_title(
            f"{name.replace('_', ' ').title()} mesh "
            f"({resolution_label})\n"
            f"{num_nodes} nodes, "
            f"{num_triangles} triangles"
        )

        single_ax.set_xlabel("x")
        single_ax.set_ylabel("y")

        # Save the individual figure without overwriting
        # figures generated at other resolutions.
        fig_single.tight_layout()

        fig_single.savefig(
            OUT / f"mesh_{name}_{suffix}.png",
            dpi=200
        )

        plt.close(fig_single)

    # Add a title identifying the gallery resolution.
    fig.suptitle(
        f"Equal-Area FEM Meshes (nominal h={spacing:g})",
        fontsize=14
    )

    # Save the combined gallery using its mesh spacing.
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    fig.savefig(
        OUT / f"mesh_gallery_{suffix}.png",
        dpi=200
    )

    plt.close(fig)

    return results


if __name__ == "__main__":

    # Generate coarse meshes to illustrate the FEM triangles.
    print("\nCoarse mesh gallery (h=0.08)")

    for result in generate_gallery(
        spacing=0.08,
        n_arc=80
    ):
        print(*result)

    # Generate fine meshes for the primary FEM resolution.
    print("\nFine mesh gallery (h=0.02)")

    for result in generate_gallery(
        spacing=0.02,
        n_arc=80
    ):
        print(*result)

    print("\nAll mesh figures saved to:")
    print(OUT)
