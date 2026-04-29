import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt

# ============================================================
# 1. Problem setup
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Geometry (quarter annulus)
r_i = 0.0015      # inner radius (m)
t   = 0.0002      # wall thickness (m)
r_o = r_i + t     # outer radius (m)

# Material properties (linear elastic)
E   = 5e5         # Young's modulus (Pa)
nu  = 0.45        # Poisson ratio

# Pressure loading
p_i = 100 * 133.322   # internal pressure (Pa)
p_o = 10  * 133.322   # external pressure (Pa)

# Lamé parameters (plane strain)
mu  = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))

# ============================================================
# 2. Analytical Lamé solution (radial displacement)
# ============================================================

def lame_constants_from_pressures(ri, ro, pi, po):
    B = (pi - po) / (1.0 / ri**2 - 1.0 / ro**2)
    A = -po + B / ro**2
    return A, B

A_lame, B_lame = lame_constants_from_pressures(r_i, r_o, p_i, p_o)

def u_r_exact(r):
    r = np.asarray(r)
    return ((1.0 + nu) / E) * ((1.0 - 2.0*nu) * A_lame * r + B_lame / r)

# ============================================================
# 3. Structured grid in quarter annulus
# ============================================================

Nr = 15
Ntheta = 15

r_vals = np.linspace(r_i, r_o, Nr)
theta_vals = np.linspace(0.0, 0.5 * math.pi, Ntheta)

R, TH = np.meshgrid(r_vals, theta_vals, indexing="ij")

x1 = (R * np.cos(TH)).reshape(-1)
x2 = (R * np.sin(TH)).reshape(-1)

x_all = np.stack([x1, x2], axis=1)
N_all = x_all.shape[0]

r_all = np.sqrt(x_all[:, 0]**2 + x_all[:, 1]**2)

# Identify boundaries and interior
inner_mask = np.isclose(r_all, r_i, atol=1e-6)
outer_mask = np.isclose(r_all, r_o, atol=1e-6)

x1_all = x_all[:, 0]
x2_all = x_all[:, 1]

# Symmetry boundaries (excluding inner radius)
x1_axis_mask = np.isclose(x2_all, 0.0, atol=1e-8) & (~inner_mask)
x2_axis_mask = np.isclose(x1_all, 0.0, atol=1e-8) & (~inner_mask)

interior_mask = ~(inner_mask | outer_mask | x1_axis_mask | x2_axis_mask)

# Split point sets
x_int_np  = x_all[interior_mask, :]
x_inn_np  = x_all[inner_mask, :]
x_out_np  = x_all[outer_mask, :]
x_x1_np   = x_all[x1_axis_mask, :]
x_x2_np   = x_all[x2_axis_mask, :]

print(f"Total points      : {N_all}")
print(f"Interior points   : {x_int_np.shape[0]}")
print(f"Inner boundary    : {x_inn_np.shape[0]}")
print(f"Outer boundary    : {x_out_np.shape[0]}")
print(f"x2 = 0 axis points: {x_x1_np.shape[0]}")
print(f"x1 = 0 axis points: {x_x2_np.shape[0]}")

# Analytical displacement for scaling
r_inn = np.sqrt(x_inn_np[:, 0]**2 + x_inn_np[:, 1]**2)
r_out = np.sqrt(x_out_np[:, 0]**2 + x_out_np[:, 1]**2)

u_r_inn_exact = u_r_exact(r_inn)
u_r_out_exact = u_r_exact(r_out)

# ============================================================
# 4. Convert to tensors
# ============================================================

def to_tensor(x, requires_grad=False):
    return torch.tensor(x, dtype=torch.float64, requires_grad=requires_grad, device=device)

x_int  = to_tensor(x_int_np,  requires_grad=True)
x_inn  = to_tensor(x_inn_np,  requires_grad=True)
x_out  = to_tensor(x_out_np,  requires_grad=True)
x_x1   = to_tensor(x_x1_np,   requires_grad=True)
x_x2   = to_tensor(x_x2_np,   requires_grad=True)

# Characteristic displacement scale
u_scale = float(max(np.max(np.abs(u_r_inn_exact)), np.max(np.abs(u_r_out_exact))))

# ============================================================
# 5. PINN model (radial displacement-based)
# ============================================================

class LamePINN(nn.Module):
    def __init__(self, nr_hidden=4, nr_neurons=40):
        super().__init__()
        self.input_layer = nn.Linear(1, nr_neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(nr_neurons, nr_neurons) for _ in range(nr_hidden)])
        self.output_layer = nn.Linear(nr_neurons, 1)
        self.activation = nn.Tanh()
        self.double()

    def forward(self, x):
        # Convert Cartesian to radius
        if x.shape[1] == 2:
            x1 = x[:, 0:1]
            x2 = x[:, 1:2]
            r = torch.sqrt(x1**2 + x2**2)
        else:
            r = x
            x1 = r
            x2 = torch.zeros_like(r)

        # Normalize radius to [-1, 1]
        r_norm = 2.0 * (r - r_i) / (r_o - r_i) - 1.0

        z = self.activation(self.input_layer(r_norm))
        for layer in self.hidden_layers:
            z = self.activation(layer(z))

        # Network predicts radial displacement
        u_r = self.output_layer(z)

        # Convert back to Cartesian components
        eps = 1e-12
        nx = x1 / (r + eps)
        ny = x2 / (r + eps)

        u1 = u_r * nx
        u2 = u_r * ny
        return torch.cat([u1, u2], dim=1)

model = LamePINN().to(device)