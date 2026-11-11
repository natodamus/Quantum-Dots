# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 21:52:02 2026

@author: natoo
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from stadium_solver import solve_stadium

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / 'results' / 'figures'
DATA = ROOT / 'results' / 'data'


def main(h=0.02, num_eigenvalues=20, num_plots=6):
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    radius = 1.0 / np.sqrt(4.0 + np.pi)
    nodes, triangles, boundary, interior, energies, eigenvectors = solve_stadium(
        h=h, a=radius, R=radius, n_arc=100,
        num_eigenvalues=num_eigenvalues
    )
    print(f'Full stadium: {len(nodes)} nodes, {len(triangles)} triangles')
    print(f'Area (analytic): {4*radius**2 + np.pi*radius**2:.8f}')
    for i, energy in enumerate(energies, start=1):
        print(f'{i:3d}: E = {energy:.8f}')
    with (DATA / 'stadium_full_energies.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['state', 'energy', 'h'])
        writer.writerows((i, float(e), h) for i, e in enumerate(energies, 1))

    triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.triplot(triangulation, color='navy', lw=0.3)
    ax.set(aspect='equal', title=f'Full stadium mesh (h={h})', xlabel='x', ylabel='y')
    fig.tight_layout()
    fig.savefig(FIGURES / 'stadium_full_mesh.png', dpi=220)
    plt.close(fig)

    count = min(num_plots, len(energies))
    fig, axes = plt.subplots((count + 1) // 2, 2, figsize=(11, 3.5 * ((count + 1) // 2)), squeeze=False)
    for j, ax in enumerate(axes.flat):
        if j >= count:
            ax.axis('off')
            continue
        psi = np.zeros(len(nodes))
        psi[interior] = eigenvectors[:, j]
        # Eigenvector sign is arbitrary. Normalize for display only.
        max_abs = np.max(np.abs(psi))
        if max_abs > 0:
            psi /= max_abs
        im = ax.tripcolor(triangulation, psi, shading='gouraud', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set(aspect='equal', title=f'State {j+1}: E={energies[j]:.5f}', xlabel='x', ylabel='y')
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle('Full-stadium eigenfunctions (no symmetry reduction)')
    fig.tight_layout()
    fig.savefig(FIGURES / 'stadium_full_eigenfunctions.png', dpi=220)
    plt.close(fig)
    print('Saved figures and CSV to results/')


if __name__ == '__main__':
    main()