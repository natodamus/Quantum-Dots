# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 23:01:47 2026

@author: natoo
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from stadium_solver import solve_stadium
from stadium_sector_solver import solve_stadium_sector

# Set the directory where the comparison figure will be saved
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'figures'
# Choose one eigenfunction from each symmetry sector
# Entries are: full-stadium state, parity sector, sector-local state
MODES = [(1, '++', 1), (2, '-+', 1), (4, '+-', 1), (5, '--', 1)]


# Restore zero values at the physical Dirichlet boundary nodes
def expand_mode(n_nodes, free_nodes, eigenvectors, column):
    values = np.zeros(n_nodes)
    values[free_nodes] = eigenvectors[:, column]
    return values


# Reflect the quarter-domain mode for comparison with the direct full-domain mode
def parity_interpolate(quarter_nodes, quarter_elements, quarter_values, full_nodes, sector):
    """Reflect quarter coordinates into Q1, interpolate, then apply parity signs."""
    # Construct an interpolator on the quarter-stadium triangles
    tri = mtri.Triangulation(quarter_nodes[:, 0], quarter_nodes[:, 1], quarter_elements)
    interpolator = mtri.LinearTriInterpolator(tri, quarter_values)
    # Map every full-stadium node to its corresponding point in the first quadrant
    x, y = full_nodes.T
    sampled = interpolator(np.abs(x), np.abs(y))
    result = np.asarray(np.ma.filled(sampled, np.nan), dtype=float)
    # Apply sign changes across axes for odd-parity eigenfunctions
    if sector[0] == '-':
        result *= np.where(x < 0, -1.0, 1.0)
    if sector[1] == '-':
        result *= np.where(y < 0, -1.0, 1.0)
    return result


# Compare full-stadium eigenfunctions with the four parity-sector solutions
def main(h=0.02, n_arc=80):
    OUT.mkdir(parents=True, exist_ok=True)
    # Set a = R to give the complete stadium an analytic area of one
    a = 1.0 / np.sqrt(4.0 + np.pi)
    print(f'Full stadium and sector eigenfunctions, h={h:g}, n_arc={n_arc}')
    # Solve on the complete stadium mesh, without symmetry boundaries
    fn, fe, fb, fi, fenergy, fv = solve_stadium(
        h=h, a=a, R=a, n_arc=n_arc, num_eigenvalues=max(m[0] for m in MODES)
    )
    # Make three plots for each mode: direct, reflected sector, and difference
    fig, axes = plt.subplots(len(MODES), 3, figsize=(13, 3.25 * len(MODES)),
                             constrained_layout=True)
    for row, (global_index, sector, local_index) in enumerate(MODES):
        # Independently solve the corresponding quarter-stadium sector
        qn, qe, qd, qfree, qenergy, qv, diagnostics = solve_stadium_sector(
            sector=sector, h=h, a=a, R=a, n_arc=n_arc,
            num_eigenvalues=local_index
        )
        # Restore boundary values and interpolate the sector mode onto full-mesh nodes
        full = expand_mode(len(fn), fi, fv, global_index - 1)
        quarter = expand_mode(len(qn), qfree, qv, local_index - 1)
        mirrored = parity_interpolate(qn, qe, quarter, fn, sector)
        # Ignore points not covered by the quarter-mesh interpolator
        valid = np.isfinite(mirrored)
        if valid.sum() < 0.8 * len(fn):
            raise RuntimeError('Too many full-mesh nodes fall outside the quarter mesh.')
        # Eigenvectors have arbitrary global signs and distinct FEM normalizations.
        # Fit one scalar on common nodes to compare their spatial shapes.
        denom = np.dot(mirrored[valid], mirrored[valid])
        scale = np.dot(full[valid], mirrored[valid]) / denom
        aligned = scale * mirrored
        # Compute the pointwise difference after sign and amplitude alignment
        difference = np.full_like(full, np.nan)
        difference[valid] = full[valid] - aligned[valid]
        relative_rms = np.linalg.norm(difference[valid]) / np.linalg.norm(full[valid])
        # Use the same color scale for both eigenfunction plots
        bound = max(np.max(np.abs(full)), np.nanmax(np.abs(aligned)))
        for col, (values, title) in enumerate([
            (full, 'Direct full-domain FEM'),
            (aligned, 'Reflected quarter-sector FEM'),
            (difference, 'Difference (full - sector)')
        ]):
            ax = axes[row, col]
            if col == 2:
                v = np.nanmax(np.abs(difference))
            else:
                v = bound
            plot = ax.tripcolor(fn[:, 0], fn[:, 1], fe, values,
                                shading='gouraud', cmap='RdBu_r', vmin=-v, vmax=v)
            ax.set_aspect('equal')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'State {global_index} ({sector}) | {title}', fontsize=10)
            fig.colorbar(plot, ax=ax, shrink=0.8)
        print(f'State {global_index:2d} ({sector}): full E={fenergy[global_index-1]:.8f}, '
              f'sector E={qenergy[local_index-1]:.8f}, '
              f'pointwise relative RMS={relative_rms:.3%}')
    # Save the completed figure for the research report
    output = OUT / 'stadium_eigenfunction_comparison.png'
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f'Saved {output}')
    print('Note: RMS is interpolation-based, not an FEM energy error estimate.')


# Read optional mesh settings from the command line
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h', type=float, default=0.02)
    parser.add_argument('--n-arc', type=int, default=80)
    args = parser.parse_args()
    main(h=args.h, n_arc=args.n_arc)