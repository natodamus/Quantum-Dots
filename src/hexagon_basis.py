# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 22:38:12 2026

@author: natoo
"""

import numpy as np

from box_basis import solve_box_basis


# Equal-area regular hexagon
R = np.sqrt(2.0 / (3.0 * np.sqrt(3.0)))
HEIGHT = np.sqrt(3.0) * R

# Leave space around the hexagon for the finite potential barrier
MARGIN = 0.10

XMIN = -R - MARGIN
XMAX = R + MARGIN
YMIN = -HEIGHT / 2.0 - MARGIN
YMAX = HEIGHT / 2.0 + MARGIN


# Identify points inside the regular hexagon
def inside_hexagon(X, Y):
    ax = np.abs(X)
    ay = np.abs(Y)

    return (
        (ax <= R)
        & (ay <= np.sqrt(3.0) * R / 2.0)
        & (np.sqrt(3.0) * ax + ay <= np.sqrt(3.0) * R)
    )


# Solve the hexagon using the finite-barrier sine basis
def solve_hexagon_basis(
    M=20,
    grid_size=220,
    V0=1e4,
    num_states=20
):
    return solve_box_basis(
        inside_function=inside_hexagon,
        xmin=XMIN,
        xmax=XMAX,
        ymin=YMIN,
        ymax=YMAX,
        mx=M,
        ny=M,
        grid_nx=grid_size,
        grid_ny=grid_size,
        V0=V0,
        num_states=num_states
    )