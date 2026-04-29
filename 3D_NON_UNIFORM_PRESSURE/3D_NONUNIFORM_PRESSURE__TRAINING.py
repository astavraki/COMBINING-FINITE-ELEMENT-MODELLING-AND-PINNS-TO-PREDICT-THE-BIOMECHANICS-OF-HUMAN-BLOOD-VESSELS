# ============================================================
# PINN training — 3D pipe with non-uniform internal pressure
# ============================================================

# Governing equations:
#   Navier–Cauchy (linear elasticity, no body forces)
#   (λ + μ) ∇(∇·u) + μ ∇²u = 0

# Boundary conditions:
#   Inner wall (r = r_i):
#       σ·n = -p_i(z)·n
#       p_i(z) varies along z (cosine distribution)

#   Outer wall (r = r_o):
#       σ·n = -p_o·n  (constant)

#   Ends (z = 0 and z = L):
#       u = 0 (fixed ends)

# Hard constraint:
#   φ(z) = z(L − z)/(L/2)^2
#   → enforces zero displacement at both ends

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math, json, csv
import matplotlib.pyplot as plt

# ============================================================
# 1. Physical parameters
# ============================================================

r_i     = 0.0015
t       = 0.0002
L       = 0.02
E       = 5e5
nu      = 0.45

p_i_mid = 100 * 133.322
p_o     = 10  * 133.322

# Collocation points
N_interior     = 1800
N_wall         = 192
N_end          = 112
resample_every = 500

# Training setup
num_epochs = 8000
bc_switch  = 1500

# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

r_o = r_i + t
mu  = E / (2*(1+nu))
lam = E*nu / ((1+nu)*(1-2*nu))

# Internal pressure distribution
def p_i_of_z(z):
    z = np.asarray(z, dtype=float)
    return p_i_mid * 0.5 * (1.0 - np.cos(2.0 * np.pi * z / L))

def p_i_of_z_t(z):
    return p_i_mid * 0.5 * (1.0 - torch.cos(2.0 * np.pi * z / L))

# Scaling
p_i_max    = float(p_i_mid)
pde_scale  = p_i_max / r_i
p_bc_scale = max(p_i_max, 1.0)

# Displacement scaling
u_scale = 3.0e-3

print(f"Device      : {device}")
print(f"mu={mu:.3e} Pa   lam={lam:.3e} Pa")
print(f"Non-uniform internal pressure (max at mid-length)")
print(f"External pressure p_o={p_o:.3e} Pa")
print(f"u_scale={u_scale:.3e} m")

# ============================================================
# 2. Collocation sampling
# ============================================================

def to_tensor(arr, grad=False):
    return torch.tensor(arr, dtype=torch.float64, requires_grad=grad, device=device)

def sample_interior(n, seed=None):
    g  = np.random.default_rng(seed)
    r  = np.sqrt(g.uniform(r_i**2, r_o**2, n))
    th = g.uniform(0, 2*math.pi, n)
    z  = g.uniform(0, L, n)
    return np.stack([r*np.cos(th), r*np.sin(th), z], axis=1)

def sample_wall(n, rv, seed=None):
    g  = np.random.default_rng(seed)
    th = g.uniform(0, 2*math.pi, n)
    z  = g.uniform(0, L, n)
    return np.stack([rv*np.cos(th), rv*np.sin(th), z], axis=1)

def sample_end(n, zv, seed=None):
    g  = np.random.default_rng(seed)
    r  = np.sqrt(g.uniform(r_i**2, r_o**2, n))
    th = g.uniform(0, 2*math.pi, n)
    return np.stack([r*np.cos(th), r*np.sin(th), np.full(n, zv)], axis=1)

def make_tensors(seed=None):
    return (
        to_tensor(sample_interior(N_interior, seed), grad=True),
        to_tensor(sample_wall(N_wall, r_i, seed),    grad=True),
        to_tensor(sample_wall(N_wall, r_o, seed),    grad=True),
        to_tensor(sample_end(N_end, 0., seed),       grad=True),
        to_tensor(sample_end(N_end, L,  seed),       grad=True)
    )

x_int, x_inn, x_out, x_z0, x_zL = make_tensors(seed=0)

# ============================================================
# 3. Network
# ============================================================

class PipePINN(nn.Module):
    def __init__(self, n_hidden=8, n_neurons=100):
        super().__init__()
        layers = [nn.Linear(4, n_neurons), nn.Tanh()]
        for _ in range(n_hidden):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 3)]
        self.net = nn.Sequential(*layers)
        self.double()

    def forward(self, x):
        x1 = x[:,0:1]; x2 = x[:,1:2]; x3 = x[:,2:3]

        r     = torch.sqrt(x1**2 + x2**2) + 1e-12
        cos_t = x1 / r
        sin_t = x2 / r

        feat = torch.cat([
            2*(r-r_i)/(r_o-r_i)-1,
            cos_t, sin_t,
            2*x3/L-1
        ], dim=1)

        out = self.net(feat)

        u_r   = out[:,0:1] * u_scale
        u_th  = out[:,1:2] * u_scale
        u_z_r = out[:,2:3] * u_scale

        phi = x3 * (L - x3) / (0.5*L)**2

        u1 = (u_r*cos_t - u_th*sin_t) * phi
        u2 = (u_r*sin_t + u_th*cos_t) * phi
        u3 = u_z_r * phi

        return torch.cat([u1, u2, u3], dim=1)