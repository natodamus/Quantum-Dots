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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"


def generate_gallery(spacing=0.08):
    OUT.mkdir(parents=True, exist_ok=True)

    # All four domains have area A = 1.
    stadium_radius = 1 / np.sqrt(4 + np.pi)

    geometries = [
        (
            "square",
            square_mesh(
                max(3, round(1 / spacing)),
                length=1.0
            )
        ),
        (
            "circle",
            circle_mesh(
                spacing=spacing,
                radius=1 / np.sqrt(np.pi)
            )[:2]
        ),
        (
            "hexagon",
            hexagon_mesh(
                spacing=spacing,
                radius=np.sqrt(2 / (3 * np.sqrt(3)))
            )[:2]
        ),
        (
            "stadium_full",
            stadium_mesh(
                h=spacing,
                a=stadium_radius,
                R=stadium_radius,
                n_arc=max(80, round(3 / spacing))
            )
        )
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for ax, (name, (nodes, triangles)) in zip(
        axes.flat, geometries
    ):
        ax.triplot(
            nodes[:, 0],
            nodes[:, 1],
            triangles,
            lw=0.42,
            color="navy"
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{name.replace('_', ' ').title()} | "
            f"{len(nodes)} nodes, "
            f"{len(triangles)} triangles"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Save individual mesh figure.
        fig_single, single_ax = plt.subplots(
            figsize=(7, 6)
        )

        single_ax.triplot(
            nodes[:, 0],
            nodes[:, 1],
            triangles,
            lw=0.55,
            color="navy"
        )

        single_ax.set_aspect(
            "equal",
            adjustable="box"
        )

        single_ax.set_title(
            f"{name.replace('_', ' ').title()} "
            f"mesh (h ≈ {spacing:g})"
        )

        single_ax.set_xlabel("x")
        single_ax.set_ylabel("y")

        fig_single.tight_layout()
        fig_single.savefig(
            OUT / f"mesh_{name}.png",
            dpi=200
        )
        plt.close(fig_single)

    # Save combined four-panel figure.
    fig.tight_layout()
    fig.savefig(
        OUT / "mesh_gallery.png",
        dpi=200
    )
    plt.close(fig)

    return [
        (name, len(nodes), len(triangles))
        for name, (nodes, triangles) in geometries
    ]


if __name__ == "__main__":
    for result in generate_gallery():
        print(*result)
