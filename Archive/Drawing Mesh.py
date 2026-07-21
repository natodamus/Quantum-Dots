import numpy as np
import matplotlib.pyplot as plt
from FEM import A_mat, B_mat
from scipy.linalg import eigh

node = np.array([
    [0.0,0.0],
    [1.0,0.0],
    [1.0,1.0],
    [0.0,1.0]
])

elm = np.array([
    [0, 1, 2],
    [0, 2, 3]
])

plt.figure(figsize=(6,6))

for e in elm:
    pts = node[e]
    pts = np.vstack([pts, pts[0]])  # close triangle
    plt.plot(pts[:,0], pts[:,1], 'k-')

for i, (x, y) in enumerate(node):
    plt.plot(x, y, 'ro')
    plt.text(x+0.02, y+0.02, str(i))

plt.axis('equal')
plt.grid(True)
plt.title("FEM Mesh Example")
plt.show()

A = A_mat(node, elm)
B = B_mat(node, elm)

print("\nA Matrix")
print(A)

print("\nB Matrix")
print(B)

# ---------------------
# Solve generalized eigenvalue problem
# ---------------------

E, psi = eigh(A, B)

print("\nEigenvalues:")
print(E)

plt.show()