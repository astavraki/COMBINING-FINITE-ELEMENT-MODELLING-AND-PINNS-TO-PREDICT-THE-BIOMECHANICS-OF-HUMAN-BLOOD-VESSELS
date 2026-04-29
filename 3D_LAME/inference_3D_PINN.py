# ============================================================
# PINN inference — 3D pressurised pipe (uniform pressure case)
# ============================================================
#
# This script:
#   • Reads a Gmsh 4.1 mesh (.msh)
#   • Evaluates the trained PINN model on:
#       - mesh nodes
#       - tetrahedral element centroids
#   • Computes displacement and stress fields
#   • Exports results to CSV for visualization (ParaView / Gmsh)
#
# Usage:
#   python pinn_inference_gmsh.py \
#       --mesh   circleFace.msh \
#       --model  model_*.pt \
#       --out    inference_results.csv
#

import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# ============================================================
# 1. Physical parameters (must match training)
# ============================================================

r_i  = 0.0015
t    = 0.0002
L    = 0.02
E    = 5e5
nu   = 0.45
p_i  = 100 * 133.322
p_o  = 10  * 133.322

r_o  = r_i + t
mu   = E / (2 * (1 + nu))
lam  = E * nu / ((1 + nu) * (1 - 2 * nu))

# Characteristic displacement scale (same as training)
def lame_constants(ri, ro, pi, po):
    B = (pi - po) / (1.0/ri**2 - 1.0/ro**2)
    A = -po + B / ro**2
    return A, B

A_lame, B_lame = lame_constants(r_i, r_o, p_i, p_o)

def u_r_exact_np(r):
    r = np.asarray(r)
    return ((1 + nu) / E) * ((1 - 2*nu) * A_lame * r + B_lame / r)

u_scale = float(max(abs(u_r_exact_np(r_i)), abs(u_r_exact_np(r_o))))

# ============================================================
# 2. Network (same architecture as training)
# ============================================================

class PipePINN(nn.Module):
    def __init__(self, n_hidden=6, n_neurons=80):
        super().__init__()
        layers = [nn.Linear(4, n_neurons), nn.Tanh()]
        for _ in range(n_hidden):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 3)]
        self.net = nn.Sequential(*layers)
        self.double()

    def forward(self, x):
        x1 = x[:, 0:1]; x2 = x[:, 1:2]; x3 = x[:, 2:3]
        r     = torch.sqrt(x1**2 + x2**2) + 1e-12
        cos_t = x1 / r;  sin_t = x2 / r

        feat  = torch.cat([
            2*(r - r_i)/(r_o - r_i) - 1,
            cos_t, sin_t,
            2*x3/L - 1
        ], dim=1)

        out   = self.net(feat)

        # cylindrical outputs → Cartesian
        u_r   = out[:, 0:1] * u_scale
        u_th  = out[:, 1:2] * u_scale
        u_z_r = out[:, 2:3] * u_scale

        phi   = x3 * (L - x3) / (0.5*L)**2   # ensures u3=0 at ends

        u1 = u_r*cos_t - u_th*sin_t
        u2 = u_r*sin_t + u_th*cos_t
        u3 = u_z_r * phi

        return torch.cat([u1, u2, u3], dim=1)

# ============================================================
# 3. Stress computation
# ============================================================

def grad_vec(f, x, retain_graph=False):
    return torch.autograd.grad(
        f, x, grad_outputs=torch.ones_like(f),
        create_graph=False, retain_graph=retain_graph
    )[0]

def strains_stresses(u, x):
    u1 = u[:, 0:1]; u2 = u[:, 1:2]; u3 = u[:, 2:3]

    g1 = grad_vec(u1, x, retain_graph=True)
    g2 = grad_vec(u2, x, retain_graph=True)
    g3 = grad_vec(u3, x, retain_graph=False)

    e11 = g1[:, 0:1]; e22 = g2[:, 1:2]; e33 = g3[:, 2:3]
    e12 = 0.5*(g1[:, 1:2] + g2[:, 0:1])
    e13 = 0.5*(g1[:, 2:3] + g3[:, 0:1])
    e23 = 0.5*(g2[:, 2:3] + g3[:, 1:2])

    tr  = e11 + e22 + e33

    s11 = lam*tr + 2*mu*e11
    s22 = lam*tr + 2*mu*e22
    s33 = lam*tr + 2*mu*e33
    s12 = 2*mu*e12
    s13 = 2*mu*e13
    s23 = 2*mu*e23

    r   = torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2) + 1e-12
    c   = x[:, 0:1]/r;  s_ = x[:, 1:2]/r

    s_rr = s11*c**2  + 2*s12*c*s_  + s22*s_**2
    s_tt = s11*s_**2 - 2*s12*c*s_  + s22*c**2

    s_vm = torch.sqrt(0.5*(
        (s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2
        + 6*(s12**2 + s13**2 + s23**2)
    ))

    return s11, s22, s33, s12, s13, s23, s_rr, s_tt, s_vm, e11, e22, e33