# COMBINING-FINITE-ELEMENT-MODELLING-AND-PINNS-TO-PREDICT-THE-BIOMECHANICS-OF-HUMAN-BLOOD-VESSELS

## Overview

This repository contains the code developed for the diplomatic project:

**"COMBINING FINITE ELEMENT MODELLING AND NEURAL NETWORKS TO PREDICT THE BIOMECHANICS OF HUMAN BLOOD VESSELS"**

The objective is to solve linear elasticity problems using Physics-Informed Neural Networks (PINNs) and compare the results with finite element simulations performed in ANSYS.

---

## Repository Structure

The repository is organized by simulation case:

* `PINN_2D/` → 2D Lamé problem
* `PINN_3D/` → 3D hollow cylinder (uniform internal pressure)
* `3D_NON_UNIFORM_PRESSURE/` → 3D cylinder with pressure varying along the axial direction
* `ANEURYSM/` → Aneurysm-like geometry simulations

Each folder is self-contained and includes all required files.

---

## File Structure (per folder)

Each simulation folder contains:

* `TRAINING.py` → PINN training script
* `inference.py` → Post-processing and evaluation of results
* `.geo` file → Geometry definition (Gmsh)
* `.msh` file → Mesh used for collocation points
* MATLAB script (`.m`) → Visualization of deformation and results (for selected cases)

---

## Requirements

* Python 3.9+
* PyTorch
* NumPy
* Matplotlib
* Gmsh

Install dependencies:

```bash
pip install torch numpy matplotlib
```

---

## How to Run

Example workflow (for any case):

1. Navigate to the desired folder:

```bash
cd PINN_3D
```

2. Train the PINN model:

```bash
python TRAINING.py
```

3. Run inference:

```bash
python inference.py
```

4. (Optional) Visualize results in MATLAB:

```matlab
pinn_deformed_shape
```

---

## Notes

* All simulations are performed in SI units (meters, Pascals)
* Boundary conditions are consistent with the thesis formulation
* Meshes are generated using Gmsh and included for reproducibility



