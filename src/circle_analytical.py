# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 22:29:24 2026

@author: natoo
"""

import numpy as np
from scipy.special import jn_zeros


# Generate the analytical spectrum of a 2D infinite circular well
def exact_circle_spectrum(radius, number_of_states=30, m_max=20, n_max=20):
    levels = []

    for m in range(m_max + 1):
        zeros = jn_zeros(m, n_max)

        for n, zero in enumerate(zeros, start=1):
            energy = zero**2 / (2.0 * radius**2)

            # m = 0 is nondegenerate, while m > 0 is twofold degenerate
            levels.append((energy, m, n))

            if m > 0:
                levels.append((energy, m, n))

    levels.sort(key=lambda level: level[0])

    return levels[:number_of_states]


if __name__ == "__main__":
    # Choose the radius so the circle has the same area as a unit square
    radius = 1.0 / np.sqrt(np.pi)
    levels = exact_circle_spectrum(radius, number_of_states=30)

    print("Analytical Circular-Well Spectrum")
    print("-" * 38)
    print(
        f"{'State':>6}"
        f"{'Energy':>14}"
        f"{'m':>6}"
        f"{'n':>6}"
    )

    for state, (energy, m, n) in enumerate(levels, start=1):
        print(
            f"{state:6d}"
            f"{energy:14.6f}"
            f"{m:6d}"
            f"{n:6d}"
        )