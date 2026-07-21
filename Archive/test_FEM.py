# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:09:59 2026

@author: natoo
"""
import numpy as np
from FEM import A_mat, B_mat

np.set_printoptions(
    precision=4,
    suppress=True,
    linewidth=120
    )

node = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
])

elm = np.array([
    [0, 1, 2],
    [0, 2, 3]
])

A = A_mat(node, elm)
B = B_mat(node, elm)

print("\nA Matrix (Stiffness Matrix)")
print("-" * 50)
print(A)

print("\nB Matrix (Overlap Matrix)")
print("-" * 50)
print(B)

import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))
plt.imshow(A)
plt.colorbar()
plt.title("Stiffness Matrix A")
plt.show()

plt.figure(figsize=(6,5))
plt.imshow(B)
plt.colorbar()
plt.title("Overlap Matrix B")
plt.show()