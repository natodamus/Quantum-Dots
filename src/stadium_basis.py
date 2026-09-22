# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 22:38:59 2026

@author: natoo
"""

import numpy as np

from box_basis import solve_box_basis


# Equal-area stadium with a = R
R = 1.0 / np.sqrt(4.0 + np.pi)
A = R

# Leave space around the stadium for the finite potential barrier
MARGIN = 0.10

XMIN = -(A + R) - MARGIN
XMAX = (A + R) + MARGIN
YMIN = -R - MARGIN
YMAX = R + MARGIN


# Identify points inside the stadium
def inside_stadium(X, Y):
    rectangle = (
        (np.abs(X) <= A)
        & (np.abs(Y) <= R)
    )

    left_cap = (X + A)**2 + Y**2 <= R**2
    right_cap = (X - A)**2 + Y**2 <= R**2

    return rectangle | left_cap | right_cap


# Solve the stadium using the finite-barrier sine basis
def solve_stadium_basis(
    M=20,
    grid_size=220,
    V0=1e4,
    num_states=20
):
    return solve_box_basis(
        inside_function=inside_stadium,
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