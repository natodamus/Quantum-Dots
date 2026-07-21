# Finite Element Modeling of Two-Dimensional Quantum Dots

This repository contains Python code developed for my M.S. Physics project at the University of Massachusetts Dartmouth.

The project uses the finite element method to solve the two-dimensional time-independent Schrödinger equation for particles confined within quantum dots of different geometries.

## Project Goals

The primary goals are to:

* construct finite-element meshes for different two-dimensional geometries;
* assemble the finite-element stiffness and mass matrices;
* solve the resulting generalized eigenvalue problem;
* calculate and visualize low-energy eigenstates;
* test numerical convergence against known analytical results; and
* compare the energy spectra of equal-area quantum dots.

The geometries currently included are:

* square;
* circle; and
* regular hexagon.

## Mathematical Model

In dimensionless form, the time-independent Schrödinger equation is written as

$$
-\frac{1}{2}\nabla^2\psi = E\psi.
$$

After finite-element discretization, the equation becomes the generalized matrix eigenvalue problem

$$
A\boldsymbol{\psi} = E B\boldsymbol{\psi},
$$

where:

* (A) is the global stiffness matrix;
* (B) is the global mass matrix;
* (E) contains the energy eigenvalues; and
* (\mathbf{\psi}) contains the corresponding eigenvectors.

Linear triangular finite elements and Dirichlet boundary conditions are used.

## Current Results

The implementation has been validated using the two-dimensional infinite square well. As the mesh is refined, the calculated ground-state energy approaches the analytical value.

| Mesh Resolution | FEM Ground-State Energy | Percent Error |
| --------------: | ----------------------: | ------------: |
|               5 |               10.861103 |       10.046% |
|              10 |               10.114213 |        2.478% |
|              20 |                9.930552 |        0.618% |
|              30 |                9.896676 |        0.274% |

The project also compares the first several energy levels of square, circular, and hexagonal domains having equal areas.

## Repository Structure

```text
src/
    FEM.py
    square_solver.py
    circle_mesh.py
    circle_solver.py
    hexagon_mesh.py
    hexagon_solver.py
    compare_geometries.py

results/
    equal_area_energy_comparison.csv
    equal_area_energy_spectra.png
    equal_area_ground_state_comparison.png

archive/
    Earlier development and testing files
```

## Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/natodamus/Quantum-Dots.git
cd Quantum-Dots
pip install -r requirements.txt
```

## Running the Geometry Comparison

From the repository’s main directory, run:

```bash
python src/compare_geometries.py
```

The comparison data and figures are saved in the `results` directory.

## Project Status

This is an active graduate physics project. Additional convergence testing, eigenfunction analysis, documentation, and interpretation of the geometry-dependent spectra are still in progress.

## Author

Renato R. Silva
M.S. Physics
University of Massachusetts Dartmouth
