import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import meshio
import matplotlib.tri as mtri

# -----------------------------
# Device selection
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# PINN model (same structure as training)
# -----------------------------
class LamePINN(nn.Module):
    def __init__(self, nr_hidden=4, nr_neurons=40):
        super().__init__()
        self.input_layer = nn.Linear(1, nr_neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(nr_neurons, nr_neurons) for _ in range(nr_hidden)]
        )
        self.output_layer = nn.Linear(nr_neurons, 1)
        self.activation = nn.Tanh()
        self.double()

    def forward(self, x):
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        r = torch.sqrt(x1**2 + x2**2)

        # map radius to [-1, 1]
        r_norm = 2.0 * (r - r_i) / (r_o - r_i) - 1.0

        z = self.activation(self.input_layer(r_norm))
        for layer in self.hidden_layers:
            z = self.activation(layer(z))

        # predicted radial displacement
        u_r = self.output_layer(z)

        eps = 1e-12
        nx = x1 / (r + eps)
        ny = x2 / (r + eps)

        # convert to Cartesian components
        u1 = u_r * nx
        u2 = u_r * ny
        return torch.cat([u1, u2], dim=1)

# -----------------------------
# Load trained model
# -----------------------------
ckpt = torch.load("lame_pinn.pt", map_location=device)

r_i = float(ckpt["r_i"])
r_o = float(ckpt["r_o"])
E   = float(ckpt["E"])
nu  = float(ckpt["nu"])
p_i = float(ckpt["p_i"])
p_o = float(ckpt["p_o"])

mu  = E / (2.0 * (1.0 + nu))
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

model = LamePINN().to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()

print("Model loaded successfully")

# -----------------------------
# Analytical Lamé solution
# -----------------------------
def lame_constants_from_pressures(ri, ro, pi, po):
    B = (pi - po) / (1.0 / ri**2 - 1.0 / ro**2)
    A = -po + B / ro**2
    return A, B

A_lame, B_lame = lame_constants_from_pressures(r_i, r_o, p_i, p_o)

def u_r_exact(r):
    r = np.asarray(r)
    return ((1.0 + nu) / E) * ((1.0 - 2.0 * nu) * A_lame * r + B_lame / r)

# -----------------------------
# Prediction utilities
# -----------------------------
def predict_displacement(xy_np):
    x = torch.tensor(xy_np, dtype=torch.float64, device=device)
    with torch.no_grad():
        u = model(x).cpu().numpy()
    return u

def grad(u_comp, x):
    return torch.autograd.grad(
        u_comp, x,
        grad_outputs=torch.ones_like(u_comp),
        create_graph=True
    )[0]

# -----------------------------
# Compute von Mises stress (PINN)
# -----------------------------
def von_mises_on_points(model, xy_np):
    x = torch.tensor(xy_np, dtype=torch.float64, device=device, requires_grad=True)

    u = model(x)
    u1 = u[:, 0:1]
    u2 = u[:, 1:2]

    gu1 = grad(u1, x)
    gu2 = grad(u2, x)

    u1_x1 = gu1[:, 0:1]
    u1_x2 = gu1[:, 1:2]
    u2_x1 = gu2[:, 0:1]
    u2_x2 = gu2[:, 1:2]

    eps11 = u1_x1
    eps22 = u2_x2
    eps12 = 0.5 * (u1_x2 + u2_x1)

    tr = eps11 + eps22

    sigma11 = lam * tr + 2.0 * mu * eps11
    sigma22 = lam * tr + 2.0 * mu * eps22
    sigma12 = 2.0 * mu * eps12

    svm = torch.sqrt(sigma11**2 - sigma11 * sigma22 + sigma22**2 + 3.0 * sigma12**2)
    return svm.detach().cpu().numpy().flatten()

# -----------------------------
# Analytical von Mises stress
# -----------------------------
def von_mises_analytic_on_points(xy_np):
    x = xy_np[:, 0]
    y = xy_np[:, 1]
    r = np.sqrt(x**2 + y**2) + 1e-15

    c = x / r
    s = y / r

    sigma_rr = A_lame - B_lame / (r**2)
    sigma_tt = A_lame + B_lame / (r**2)

    sigma11 = sigma_rr * c**2 + sigma_tt * s**2
    sigma22 = sigma_rr * s**2 + sigma_tt * c**2
    sigma12 = (sigma_rr - sigma_tt) * c * s

    svm = np.sqrt(sigma11**2 - sigma11 * sigma22 + sigma22**2 + 3.0 * sigma12**2)
    return svm

# -----------------------------
# Load mesh and build triangulation
# -----------------------------
mesh = meshio.read("quarter_annulus.msh")
pts = mesh.points[:, :2]
x = pts[:, 0]
y = pts[:, 1]

tri = None
for block in mesh.cells:
    if block.type == "triangle":
        tri = block.data
        break

if tri is None:
    raise ValueError("Mesh must contain triangular elements.")

triang = mtri.Triangulation(x, y, tri)

# remove triangles outside domain (safety check)
xtri = x[triang.triangles]
ytri = y[triang.triangles]
rtri = np.sqrt(xtri**2 + ytri**2)

bad = (
    (rtri < r_i - 1e-15).any(axis=1)
    | (rtri > r_o + 1e-15).any(axis=1)
    | (xtri < -1e-15).any(axis=1)
    | (ytri < -1e-15).any(axis=1)
)
triang.set_mask(bad)

theta_plot = np.linspace(0.0, 0.5 * math.pi, 250)

# -----------------------------
# Displacement fields (analytic vs PINN)
# -----------------------------
u_pred = predict_displacement(pts)
u1_p = u_pred[:, 0]
u2_p = u_pred[:, 1]
u_mag_p = np.sqrt(u1_p**2 + u2_p**2)

r_nodes = np.sqrt(x**2 + y**2) + 1e-15
u_r_a = u_r_exact(r_nodes)
u1_a = u_r_a * (x / r_nodes)
u2_a = u_r_a * (y / r_nodes)
u_mag_a = np.sqrt(u1_a**2 + u2_a**2)

u_err = np.sqrt((u1_p - u1_a)**2 + (u2_p - u2_a)**2)