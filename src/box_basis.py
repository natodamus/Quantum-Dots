# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 22:37:16 2026

@author: natoo
"""
import numpy as np
from scipy.linalg import eigh


# Solve a 2D quantum dot using a sine basis inside a rectangular box
def solve_box_basis(
    inside_function,
    xmin,
    xmax,
    ymin,
    ymax,
    mx=12,
    ny=12,
    grid_nx=160,
    grid_ny=160,
    V0=1e5,
    num_states=30
):
    Lx = xmax - xmin
    Ly = ymax - ymin

    # Build the numerical integration grid
    x = np.linspace(xmin, xmax, grid_nx)
    y = np.linspace(ymin, ymax, grid_ny)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y, indexing="xy")

    # Set V = 0 inside the dot and V = V0 outside it
    inside = inside_function(X, Y)
    V = np.where(inside, 0.0, V0)

    # Generate the rectangular-box basis states
    basis_states = [
        (m, n)
        for m in range(1, mx + 1)
        for n in range(1, ny + 1)
    ]

    nbasis = len(basis_states)

    xf = X.ravel()
    yf = Y.ravel()
    Vf = V.ravel()

    # Evaluate every basis function on the integration grid
    Phi = np.zeros((xf.size, nbasis))
    kinetic = np.zeros(nbasis)

    for j, (m, n) in enumerate(basis_states):
        Phi[:, j] = (
            2.0 / np.sqrt(Lx * Ly)
            * np.sin(m * np.pi * (xf - xmin) / Lx)
            * np.sin(n * np.pi * (yf - ymin) / Ly)
        )

        kinetic[j] = 0.5 * (
            (m * np.pi / Lx)**2
            + (n * np.pi / Ly)**2
        )

    # The kinetic-energy operator is diagonal in the sine basis
    T = np.diag(kinetic)

    # Use 2D trapezoidal weights for the potential-energy integrals
    wx = np.ones(grid_nx)
    wy = np.ones(grid_ny)

    wx[[0, -1]] = 0.5
    wy[[0, -1]] = 0.5

    W = np.outer(wy, wx).ravel() * dx * dy

    # Vij = integral phi_i V phi_j dxdy
    weighted_phi = (W * Vf)[:, None] * Phi
    Vmat = Phi.T @ weighted_phi

    # Solve Hc = Ec
    H = T + Vmat
    energies, coefficients = eigh(H)

    return (
        energies[:num_states],
        coefficients[:, :num_states],
        basis_states
    )