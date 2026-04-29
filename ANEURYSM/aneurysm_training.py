"""
PINN training for the 3D aneurysm-like geometry.

The model solves the Navier-Cauchy equations for linear elasticity using
mesh-based collocation points. Boundary points are sampled from the Gmsh
surface mesh and use precomputed outward normals for the traction terms.

Coordinate convention:
  x, z : cross-section plane
  y    : axial direction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import json
import os

try:
    import meshio
except ImportError:
    os.system("pip install meshio --quiet --break-system-packages")
    import meshio

# ============================================================
# 1. Input parameters
# ============================================================

MESH_FILE  = "Part1.msh"   # Gmsh mesh file
MESH_SCALE = 1e-3           # unit conversion: mesh is in mm -> metres

# Material properties
E   = 5e5    # Young's modulus  [Pa]
nu  = 0.45   # Poisson's ratio  [-]

# Pressure values
p_i = 100 * 133.322   # inner wall  [Pa]  (100 mmHg)
p_o = 10  * 133.322   # outer wall  [Pa]  ( 10 mmHg)

# Collocation points
N_interior_A   = 800    # Phase A: w_eq=1 so interior barely matters — use few
N_interior_B   = 4000   # Phase B/C: PDE active — use full set
N_interior     = N_interior_A  # start with Phase A count
N_wall         = 1200   # more wall points: BC loss was plateauing with 800
N_end          = 200    # random points per end face (y=0 / y=L)
resample_every = 1000   # re-draw less often — frequent resampling destabilises training

# Training
num_epochs  = 6000
bc_switch   = 600    # Phase A shorter: hybrid inputs learn BCs faster
pde_switch  = 2000   # Phase B ends here; Phase C runs 2001-6000
                      # Phase C (2001-6000): PDE dominant, BCs gently held

# Optional ANSYS comparison file
# Required columns: x_mm,y_mm,z_mm,u_x_m,u_y_m,u_z_m,u_mag_m,sigma_vm
ANSYS_RESULTS_CSV = None   # e.g. "ansys_nodal_results.csv"

# ============================================================
# End of input parameters
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mu     = E / (2.0 * (1.0 + nu))
lam    = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

print(f"Device      : {device}")
print(f"mu={mu:.3e} Pa  lam={lam:.3e} Pa")
print(f"p_i={p_i:.3e} Pa  p_o={p_o:.3e} Pa")

# ============================================================
# 2. Mesh loading and surface normals
# ============================================================

print(f"\nLoading mesh '{MESH_FILE}' ...")
mesh = meshio.read(MESH_FILE)
pts  = mesh.points * MESH_SCALE           # (N_nodes, 3) in metres

L     = pts[:, 1].max()                   # axial length along y-axis
r_max = float(np.sqrt(pts[:, 0]**2 + pts[:, 2]**2).max())

print(f"  Nodes      : {len(pts)}")
print(f"  Length L   : {L*1e3:.3f} mm   (y-axis)")
print(f"  r_max      : {r_max*1e3:.3f} mm")

# Element groups
tets     = next(cb.data for cb in mesh.cells if cb.type == "tetra")
all_tris = np.vstack([cb.data for cb in mesh.cells if cb.type == "triangle"])
print(f"  Tetrahedra : {len(tets)}")
print(f"  Triangles  : {len(all_tris)}")

# Surface classification
tri_cents = pts[all_tris].mean(axis=1)   # (N_tri, 3)
tol_y     = 1e-7                          # metres

is_end0 = tri_cents[:, 1] < tol_y
is_endL = np.abs(tri_cents[:, 1] - L) < tol_y
is_wall = ~(is_end0 | is_endL)

# Outward normals from tetrahedral connectivity
# Each wall triangle is matched to its adjacent tetrahedron.
# The opposite tetrahedral vertex is used to set the outward direction.
print("  Computing outward normals (tet-face adjacency) ...")
local_faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
tet_face_map = {}
for tet in tets:
    for lf in local_faces:
        key = frozenset(tet[list(lf)])
        opp = next(tet[j] for j in range(4) if j not in lf)
        tet_face_map[key] = opp

wall_tris    = all_tris[is_wall]
wall_normals = np.zeros((len(wall_tris), 3))
for k, tri in enumerate(wall_tris):
    opp     = tet_face_map[frozenset(tri)]
    v0, v1, v2 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
    n = np.cross(v1 - v0, v2 - v0).astype(float)
    if np.dot(n, v0 - pts[opp]) < 0:
        n = -n
    wall_normals[k] = n / (np.linalg.norm(n) + 1e-15)

# Inner and outer wall separation
wc    = tri_cents[is_wall]
r_wc  = np.sqrt(wc[:, 0]**2 + wc[:, 2]**2) + 1e-15
r_hat = np.column_stack([wc[:, 0]/r_wc, np.zeros(len(wc)), wc[:, 2]/r_wc])
inner_wall_mask = (wall_normals * r_hat).sum(axis=1) < 0
outer_wall_mask = ~inner_wall_mask

inn_tris    = wall_tris[inner_wall_mask]
out_tris    = wall_tris[outer_wall_mask]
inn_normals = wall_normals[inner_wall_mask]
out_normals = wall_normals[outer_wall_mask]

end0_tris = all_tris[is_end0]
endL_tris = all_tris[is_endL]

print(f"  Inner wall : {len(inn_tris):5d} tris  "
      f"(mean r = {r_wc[inner_wall_mask].mean()*1e3:.3f} mm)")
print(f"  Outer wall : {len(out_tris):5d} tris  "
      f"(mean r = {r_wc[outer_wall_mask].mean()*1e3:.3f} mm)")
print(f"  End y=0    : {is_end0.sum():5d} tris")
print(f"  End y=L    : {is_endL.sum():5d} tris")

# Displacement and PDE scaling
# The estimated displacement scale is used only for numerical scaling.
r_i_est = r_wc[inner_wall_mask].mean()
r_o_est = r_wc[outer_wall_mask].mean()
t_est   = r_o_est - r_i_est
B_s     = (p_i - p_o) / (1.0/r_i_est**2 - 1.0/r_o_est**2)
A_s     = -p_o + B_s / r_o_est**2
u_r_est = lambda r: ((1+nu)/E) * ((1-2*nu)*A_s*r + B_s/r)
u_scale   = float(max(abs(u_r_est(r_i_est)), abs(u_r_est(r_o_est))))
u_scale = 1.5 * u_scale
pde_scale = 0.5*(p_i / t_est)
print(f"\nu_scale={u_scale:.3e} m   pde_scale={pde_scale:.3e} Pa/m")

# ============================================================
# 3. Mesh-based collocation sampling
# ============================================================

def sample_volume(n, seed=None):
    """Sample random interior points from tetrahedral elements."""
    g   = np.random.default_rng(seed)
    idx = g.integers(0, len(tets), n)
    bary = g.dirichlet([1, 1, 1, 1], size=n)
    return (bary[:, 0:1]*pts[tets[idx, 0]] +
            bary[:, 1:2]*pts[tets[idx, 1]] +
            bary[:, 2:3]*pts[tets[idx, 2]] +
            bary[:, 3:4]*pts[tets[idx, 3]])


def sample_surface(n, tri_idx_arr, tri_norm_arr, seed=None):
    """Sample random points from surface triangles and return their normals."""
    g      = np.random.default_rng(seed)
    chosen = g.integers(0, len(tri_idx_arr), n)
    u, v   = g.random(n), g.random(n)
    bad    = u + v > 1
    u[bad] = 1 - u[bad]
    v[bad] = 1 - v[bad]
    w      = 1 - u - v
    tris   = tri_idx_arr[chosen]
    pts_s  = (w[:, None]*pts[tris[:, 0]] +
              u[:, None]*pts[tris[:, 1]] +
              v[:, None]*pts[tris[:, 2]])
    return pts_s, tri_norm_arr[chosen].copy()


def to_tensor(arr, grad=False):
    return torch.tensor(arr, dtype=torch.float32,
                        requires_grad=grad, device=device)


def make_tensors(seed=None):
    """Create the current collocation and boundary point tensors."""
    x_int_np           = sample_volume(N_interior, seed)
    x_inn_np, n_inn_np = sample_surface(N_wall, inn_tris, inn_normals, seed)
    x_out_np, n_out_np = sample_surface(N_wall, out_tris, out_normals, seed)
    x_z0_np, _         = sample_surface(N_end,  end0_tris,
                                         np.zeros((len(end0_tris), 3)), seed)
    x_zL_np, _         = sample_surface(N_end,  endL_tris,
                                         np.zeros((len(endL_tris), 3)), seed)
    return (to_tensor(x_int_np, grad=True),
            to_tensor(x_inn_np, grad=True), to_tensor(n_inn_np),
            to_tensor(x_out_np, grad=True), to_tensor(n_out_np),
            to_tensor(x_z0_np,  grad=True),
            to_tensor(x_zL_np,  grad=True))


x_int, x_inn, n_inn, x_out, n_out, x_z0, x_zL = make_tensors(seed=0)

# Fixed wall points used in the traction loss
# These points are kept constant during training.
x_inn_dense_np, n_inn_dense_np = sample_surface(N_wall, inn_tris, inn_normals, seed=99)
x_out_dense_np, n_out_dense_np = sample_surface(N_wall, out_tris, out_normals, seed=100)
x_inn_dense = to_tensor(x_inn_dense_np, grad=True);  n_inn_dense = to_tensor(n_inn_dense_np)
x_out_dense = to_tensor(x_out_dense_np, grad=True);  n_out_dense = to_tensor(n_out_dense_np)

print(f"\nRandom sampling:")
print(f"  Interior pts  : {N_interior}")
print(f"  Inner wall    : {N_wall}  (+ {N_wall} fixed dense)")
print(f"  Outer wall    : {N_wall}  (+ {N_wall} fixed dense)")
print(f"  End caps      : {N_end} each")
print(f"  Resample every: {resample_every} epochs")

# ============================================================
# 4. Network
#
# Inputs: r/r_max, 2y/L-1, cos(theta), sin(theta)
# Output components are converted to Cartesian displacement.
# Axial displacement uses a hard constraint at the two ends.
#
# Hard constraint: phi(y)=y(L-y)/(L/2)^2
#
#
#
#
# ============================================================

class AneurysmPINN(nn.Module):
    """PINN with cylindrical features and Cartesian displacement output."""
    def __init__(self, n_hidden: int = 6, n_neurons: int = 80):
        super().__init__()
        layers = [nn.Linear(4, n_neurons), nn.Tanh()]   # 4 inputs, not 3
        for _ in range(n_hidden):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 3)]
        self.net = nn.Sequential(*layers)
        self.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x[:, 0:1]   # Cartesian x  (cross-section)
        y_c = x[:, 1:2]   # Cartesian y  (axial, 0->L)
        z_c = x[:, 2:3]   # Cartesian z  (cross-section)

        r     = torch.sqrt(x_c**2 + z_c**2) + 1e-12
        cos_t = x_c / r    # cos(theta) -- encodes circumferential direction
        sin_t = z_c / r    # sin(theta)
        r_n   = r / r_max
        y_n   = 2.0 * y_c / L - 1.0

        feat = torch.cat([r_n, y_n, cos_t, sin_t], dim=1)  # (N, 4)
        out  = self.net(feat)

        # Cylindrical displacement components
        u_r_raw = out[:, 0:1] * u_scale   # radial
        u_y_raw = out[:, 1:2] * u_scale   # axial
        u_t_raw = out[:, 2:3] * u_scale   # circumferential (small for axisymm.)

        # Axial hard constraint
        phi = y_c * (L - y_c) / (0.5 * L)**2

        # Cylindrical to Cartesian displacement
        u_x = u_r_raw * cos_t - u_t_raw * sin_t
        u_y = u_y_raw * phi
        u_z = u_r_raw * sin_t + u_t_raw * cos_t

        return torch.cat([u_x, u_y, u_z], dim=1)


model = AneurysmPINN(n_hidden=8, n_neurons=100).to(device)
n_p = sum(p.numel() for p in model.parameters())
print(f"\nParameters: {n_p}")

# ============================================================
# 5. Autograd helpers
# ============================================================

def grad_vec(f, x):
    return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f),
                               create_graph=True)[0]

def diag_hessian(g_col, x):
    return torch.autograd.grad(g_col, x, grad_outputs=torch.ones_like(g_col),
                               create_graph=True)[0]

def laplacian_from_grads(g1, g2, g3, x):
    """Laplacian terms from first-order gradients."""
    lap1 = (diag_hessian(g1[:,0:1],x)[:,0:1] +
            diag_hessian(g1[:,1:2],x)[:,1:2] +
            diag_hessian(g1[:,2:3],x)[:,2:3])
    lap2 = (diag_hessian(g2[:,0:1],x)[:,0:1] +
            diag_hessian(g2[:,1:2],x)[:,1:2] +
            diag_hessian(g2[:,2:3],x)[:,2:3])
    lap3 = (diag_hessian(g3[:,0:1],x)[:,0:1] +
            diag_hessian(g3[:,1:2],x)[:,1:2] +
            diag_hessian(g3[:,2:3],x)[:,2:3])
    return lap1, lap2, lap3

def strains_stresses(u, x):
    """Cartesian stress components and von Mises stress."""
    u1, u2, u3 = u[:,0:1], u[:,1:2], u[:,2:3]
    g1=grad_vec(u1,x); g2=grad_vec(u2,x); g3=grad_vec(u3,x)
    e11=g1[:,0:1]; e22=g2[:,1:2]; e33=g3[:,2:3]
    e12=0.5*(g1[:,1:2]+g2[:,0:1])
    e13=0.5*(g1[:,2:3]+g3[:,0:1])
    e23=0.5*(g2[:,2:3]+g3[:,1:2])
    tr=e11+e22+e33
    s11=lam*tr+2*mu*e11; s22=lam*tr+2*mu*e22; s33=lam*tr+2*mu*e33
    s12=2*mu*e12; s13=2*mu*e13; s23=2*mu*e23
    s_vm=torch.sqrt(0.5*((s11-s22)**2+(s22-s33)**2+(s33-s11)**2
                         +6*(s12**2+s13**2+s23**2)))
    return s11,s22,s33,s12,s13,s23,s_vm

def traction_residual(mdl, x_pts, normals_t, p_target):
    """Traction residual on a boundary with known outward normals."""
    if not x_pts.requires_grad:
        x_pts = x_pts.clone().detach().requires_grad_(True)
    u  = mdl(x_pts)
    s11,s22,s33,s12,s13,s23,_ = strains_stresses(u, x_pts)
    n1,n2,n3 = normals_t[:,0:1], normals_t[:,1:2], normals_t[:,2:3]
    p = torch.tensor(p_target, dtype=torch.float32, device=device)
    r1 = (s11*n1 + s12*n2 + s13*n3) + p*n1
    r2 = (s12*n1 + s22*n2 + s23*n3) + p*n2
    r3 = (s13*n1 + s23*n2 + s33*n3) + p*n3
    return r1, r2, r3

# ============================================================
# 6. Loss function
#
# ============================================================

# Training weights
# Phase A: traction terms dominate.
# Phase B: PDE and traction terms are balanced.
# Phase C: PDE residual is weighted more strongly.
# The traction loss remains active throughout training.
#
#
w_bc_global = 200.0
w_eq_global = 1.0

def loss_pinn(mdl):
    # PDE residual
    x=x_int; u=mdl(x)
    u1=u[:,0:1]; u2=u[:,1:2]; u3=u[:,2:3]
    g1=grad_vec(u1,x); g2=grad_vec(u2,x); g3=grad_vec(u3,x)
    div_u   =g1[:,0:1]+g2[:,1:2]+g3[:,2:3]
    grad_div=grad_vec(div_u,x)
    lap1,lap2,lap3=laplacian_from_grads(g1,g2,g3,x)
    F1=(lam+mu)*grad_div[:,0:1]+mu*lap1
    F2=(lam+mu)*grad_div[:,1:2]+mu*lap2
    F3=(lam+mu)*grad_div[:,2:3]+mu*lap3
    loss_eq=torch.mean((F1/pde_scale)**2+(F2/pde_scale)**2+(F3/pde_scale)**2)

    # Traction boundary conditions
    r1,r2,r3 = traction_residual(mdl, x_inn_dense, n_inn_dense, p_i)
    loss_inn  = torch.mean((r1/p_i)**2+(r2/p_i)**2+(r3/p_i)**2)
    r1,r2,r3 = traction_residual(mdl, x_out_dense, n_out_dense, p_o)
    loss_out  = torch.mean((r1/p_i)**2+(r2/p_i)**2+(r3/p_i)**2)

    # End-face displacement penalty
    loss_z0 = torch.mean((mdl(x_z0)/u_scale)**2)
    loss_zL = torch.mean((mdl(x_zL)/u_scale)**2)

    # Phase-dependent weights
    #
    total = (w_eq_global   * loss_eq
           + w_bc_global   * (loss_inn + loss_out)
           + 10.0          * (loss_z0  + loss_zL))

    comp = dict(eq=loss_eq.item(), inn=loss_inn.item(), out=loss_out.item(),
                z0=loss_z0.item(), zL=loss_zL.item())
    return total, comp

# ============================================================
# 7. Training
# Phase A, Phase B and Phase C use different weights and learning rates.
#
# L-BFGS is applied after Adam optimization.
# ============================================================

history = {k: [] for k in ["loss","eq","inn","out","z0","zL"]}

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=300, verbose=True)

print(f"\n{'='*60}")
print(f"Phase A: Adam  w_bc={w_bc_global:.0f}  w_eq={w_eq_global:.0f}  epochs 1-{bc_switch}")
print(f"Phase B will start at epoch {bc_switch+1}: w_bc=50  w_eq=50")
print(f"Phase C will start at epoch {pde_switch+1}: w_bc=30  w_eq=150")
print(f"{'='*60}")

for epoch in range(1, num_epochs+1):

    if epoch == bc_switch+1:
        w_bc_global = 50.0
        w_eq_global = 50.0
        N_interior  = N_interior_B  # switch to full interior set now PDE is active
        x_int, x_inn, n_inn, x_out, n_out, x_z0, x_zL = make_tensors(seed=epoch)
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=400, verbose=True)
        print(f"\n{'='*60}")
        print(f"Phase B: Adam  w_bc=50  w_eq=50  N_int={N_interior}  epochs {bc_switch+1}-{pde_switch}")
        print(f"{'='*60}")

    if epoch == pde_switch+1:
        w_bc_global = 50.0   # Keep BCs firmly — prevents amplitude drift
        w_eq_global = 150.0  # PDE strongly dominates
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=500, verbose=True)
        print(f"\n{'='*60}")
        print(f"Phase C: Adam  w_bc=50  w_eq=150  epochs {pde_switch+1}-{num_epochs}")
        print(f"{'='*60}")

    # Periodic resampling of collocation points
    if epoch % resample_every == 0:
        x_int, x_inn, n_inn, x_out, n_out, x_z0, x_zL = make_tensors(seed=epoch)

    optimizer.zero_grad()
    loss, comp = loss_pinn(model)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step(loss)

    if not torch.isfinite(loss):
        print(f"NaN/Inf at epoch {epoch}"); break

    history["loss"].append(loss.item())
    for k in comp: history[k].append(comp[k])

    if epoch % 500 == 0 or epoch == 1:
        print(f"Ep {epoch:5d}  total={loss.item():.3e}  "
              f"eq={comp['eq']:.3e}  bc={comp['inn']+comp['out']:.3e}  "
              f"w_eq={w_eq_global:.0f}  w_bc={w_bc_global:.0f}")

# L-BFGS refinement
print(f"\n{'='*60}")
print("Phase C: L-BFGS polish")
print(f"{'='*60}")
w_bc_global = 50.0
w_eq_global = 150.0
opt_lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1000,
                            tolerance_grad=1e-13, tolerance_change=1e-13,
                            history_size=100, line_search_fn='strong_wolfe')
def closure():
    opt_lb.zero_grad()
    l, _ = loss_pinn(model)
    l.backward()
    return l
opt_lb.step(closure)
print("Training complete!")

# ============================================================
# 8. Save model
# ============================================================

tag = f"aneurysm_Ni{N_interior}_Nw{N_wall}_ep{num_epochs}"
torch.save(model.state_dict(), f"model_{tag}.pt")
print(f"\nModel saved: model_{tag}.pt")

# ============================================================
# 9. Sanity checks
# ============================================================

with torch.enable_grad():
    r1i,r2i,r3i = traction_residual(model, x_inn, n_inn, p_i)
    r1o,r2o,r3o = traction_residual(model, x_out, n_out, p_o)
    err_inn = torch.max(torch.sqrt(r1i**2+r2i**2+r3i**2)).item()
    err_out = torch.max(torch.sqrt(r1o**2+r2o**2+r3o**2)).item()
with torch.no_grad():
    u_end0 = model(x_z0).abs().max().item()
    u_endL = model(x_zL).abs().max().item()

print(f"\nMax inner traction residual: {err_inn:.3e} Pa  (p_i={p_i:.3e})")
print(f"Max outer traction residual: {err_out:.3e} Pa  (p_o={p_o:.3e})")
print(f"Max |u| at y=0             : {u_end0:.3e} m")
print(f"Max |u| at y=L             : {u_endL:.3e} m")

# ============================================================
# 10. Evaluation at mesh nodes
# Batched inference is used for memory control.
# ============================================================

model.eval()
print("\nEvaluating at all mesh nodes ...")
EVAL_BATCH = 512

def strains_stresses_infer(u, x):
    """Stress calculation used during batched export."""
    u1,u2,u3 = u[:,0:1],u[:,1:2],u[:,2:3]
    g1 = torch.autograd.grad(u1, x, grad_outputs=torch.ones_like(u1),
                              create_graph=False, retain_graph=True)[0]
    g2 = torch.autograd.grad(u2, x, grad_outputs=torch.ones_like(u2),
                              create_graph=False, retain_graph=True)[0]
    g3 = torch.autograd.grad(u3, x, grad_outputs=torch.ones_like(u3),
                              create_graph=False, retain_graph=False)[0]
    e11=g1[:,0:1]; e22=g2[:,1:2]; e33=g3[:,2:3]
    e12=0.5*(g1[:,1:2]+g2[:,0:1])
    e13=0.5*(g1[:,2:3]+g3[:,0:1])
    e23=0.5*(g2[:,2:3]+g3[:,1:2])
    tr=e11+e22+e33
    s11=lam*tr+2*mu*e11; s22=lam*tr+2*mu*e22; s33=lam*tr+2*mu*e33
    s12=2*mu*e12; s13=2*mu*e13; s23=2*mu*e23
    svm=torch.sqrt(0.5*((s11-s22)**2+(s22-s33)**2+(s33-s11)**2
                        +6*(s12**2+s13**2+s23**2)))
    return s11,s22,s33,s12,s13,s23,svm

n_nodes = len(pts)
u_np   = np.zeros((n_nodes,3))
s11_np = np.zeros(n_nodes); s22_np = np.zeros(n_nodes)
s33_np = np.zeros(n_nodes); s12_np = np.zeros(n_nodes)
s13_np = np.zeros(n_nodes); s23_np = np.zeros(n_nodes)
svm_np = np.zeros(n_nodes)

for i in range(0, n_nodes, EVAL_BATCH):
    x_b = torch.tensor(pts[i:i+EVAL_BATCH], dtype=torch.float32,
                        device=device, requires_grad=True)
    with torch.enable_grad():
        u_b = model(x_b)
        s11b,s22b,s33b,s12b,s13b,s23b,svmb = strains_stresses_infer(u_b, x_b)
    j = i + len(x_b)
    u_np[i:j]   = u_b.detach().cpu().numpy()
    s11_np[i:j] = s11b.detach().cpu().numpy().ravel()
    s22_np[i:j] = s22b.detach().cpu().numpy().ravel()
    s33_np[i:j] = s33b.detach().cpu().numpy().ravel()
    s12_np[i:j] = s12b.detach().cpu().numpy().ravel()
    s13_np[i:j] = s13b.detach().cpu().numpy().ravel()
    s23_np[i:j] = s23b.detach().cpu().numpy().ravel()
    svm_np[i:j] = svmb.detach().cpu().numpy().ravel()

u_mag = np.linalg.norm(u_np, axis=1)
print(f"  |u|  range : [{u_mag.min():.3e}, {u_mag.max():.3e}] m")
print(f"  s_vm range : [{svm_np.min():.3e}, {svm_np.max():.3e}] Pa")

# ============================================================
# Plane values and contour plots
# ============================================================

try:
    from scipy.interpolate import griddata
except ImportError:
    os.system("pip install scipy --quiet --break-system-packages")
    from scipy.interpolate import griddata

plane_defs = [
    ("L/3",  L/3),
    ("L/2",  L/2),
    ("3L/4", 3*L/4),
]

# Slice thickness around each axial plane
plane_tol = 0.015 * L

print(f"\n{'='*60}")
print("  PLANE SUMMARY  (PINN)")
print(f"{'='*60}")

plane_data = []

for label, y_plane in plane_defs:
    mask = np.abs(pts[:, 1] - y_plane) < plane_tol

    if mask.sum() == 0:
        print(f"\n  At y={label}: no nodes found in slice")
        plane_data.append(None)
        continue

    pts_plane  = pts[mask]
    umag_plane = u_mag[mask]
    svm_plane  = svm_np[mask]

    print(f"\n  At y={label} = {y_plane*1e3:.3f} mm:")
    print(f"    nodes in slice = {mask.sum()}")
    print(f"    Max |u|        = {umag_plane.max():.3e} m")
    print(f"    Max σ_vm       = {svm_plane.max():.3e} Pa")

    plane_data.append((label, y_plane, pts_plane, umag_plane, svm_plane))

# Contour plots in x-z cross-sections
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    "PINN plane results at y = L/3, L/2, 3L/4\n"
    "Row 1: total deformation |u|     Row 2: von Mises stress σ_vm",
    fontsize=13
)

for col, item in enumerate(plane_data):
    if item is None:
        axes[0, col].axis("off")
        axes[1, col].axis("off")
        continue

    label, y_plane, pts_plane, umag_plane, svm_plane = item

    x_mm = pts_plane[:, 0] * 1e3
    z_mm = pts_plane[:, 2] * 1e3

    # Regular x-z grid
    xi = np.linspace(x_mm.min(), x_mm.max(), 300)
    zi = np.linspace(z_mm.min(), z_mm.max(), 300)
    XI, ZI = np.meshgrid(xi, zi)

    # Linear interpolation onto the plot grid
    U_grid = griddata((x_mm, z_mm), umag_plane, (XI, ZI), method="linear")
    S_grid = griddata((x_mm, z_mm), svm_plane,  (XI, ZI), method="linear")

    # Mask outside the available cross-section
    valid = griddata((x_mm, z_mm), np.ones_like(x_mm), (XI, ZI), method="linear")
    U_grid = np.where(np.isnan(valid), np.nan, U_grid)
    S_grid = np.where(np.isnan(valid), np.nan, S_grid)

    # Mask the hollow centre
    R = np.sqrt(XI**2 + ZI**2)
    r_pts = np.sqrt(x_mm**2 + z_mm**2)
    r_min = r_pts.min()
    r_max_slice = r_pts.max()

    mask_ring = (R >= r_min) & (R <= r_max_slice)
    U_grid[~mask_ring] = np.nan
    S_grid[~mask_ring] = np.nan

    # Displacement contour
    cf1 = axes[0, col].contourf(XI, ZI, U_grid, levels=40, cmap="viridis")
    axes[0, col].scatter(x_mm, z_mm, s=1, c="k", alpha=0.08)
    plt.colorbar(cf1, ax=axes[0, col]).set_label("|u| (m)")
    axes[0, col].set_aspect("equal")
    axes[0, col].set_xlabel("x (mm)")
    axes[0, col].set_ylabel("z (mm)")
    axes[0, col].set_title(f"|u| at y={label} = {y_plane*1e3:.2f} mm")

    # von Mises contour
    cf2 = axes[1, col].contourf(XI, ZI, S_grid, levels=40, cmap="plasma")
    axes[1, col].scatter(x_mm, z_mm, s=1, c="k", alpha=0.08)
    plt.colorbar(cf2, ax=axes[1, col]).set_label("σ_vm (Pa)")
    axes[1, col].set_aspect("equal")
    axes[1, col].set_xlabel("x (mm)")
    axes[1, col].set_ylabel("z (mm)")
    axes[1, col].set_title(f"σ_vm at y={label} = {y_plane*1e3:.2f} mm")

plt.tight_layout()
plt.savefig("aneurysm_plane_contours.png", dpi=150)
plt.show()

print("Saved -> aneurysm_plane_contours.png")



# ============================================================
# 11. Export results
# ============================================================

vtk_out = meshio.Mesh(
    points = pts,
    cells  = mesh.cells,
    point_data = {
        # VTK field names
        "Disp_X_m"      : u_np[:,0],
        "Disp_Y_m"      : u_np[:,1],
        "Disp_Z_m"      : u_np[:,2],
        "Disp_mag_m"    : u_mag,
        "Sigma_XX"      : s11_np,
        "Sigma_YY"      : s22_np,
        "Sigma_ZZ"      : s33_np,
        "Sigma_XY"      : s12_np,
        "Sigma_XZ"      : s13_np,
        "Sigma_YZ"      : s23_np,
        "Sigma_vonMises": svm_np,
    },
)
vtk_file = f"pinn_results_{tag}.vtk"
meshio.write(vtk_file, vtk_out)
print(f"\nVTK  -> {vtk_file}   (open in ParaView or import into ANSYS)")

csv_file = f"pinn_results_{tag}.csv"
np.savetxt(csv_file,
           np.column_stack([pts*1e3, u_np, u_mag[:,None],
                            s11_np[:,None], s22_np[:,None], s33_np[:,None],
                            s12_np[:,None], s13_np[:,None], s23_np[:,None],
                            svm_np[:,None]]),
           delimiter=",",
           header="x_mm,y_mm,z_mm,u_x_m,u_y_m,u_z_m,u_mag_m,"
                  "sigma_xx,sigma_yy,sigma_zz,sigma_xy,sigma_xz,sigma_yz,sigma_vm",
           comments="")
print(f"CSV  -> {csv_file}")

metrics = dict(
    N_interior=N_interior, N_wall=N_wall, N_end=N_end,
    resample_every=resample_every, num_epochs=num_epochs,
    u_mag_max_m=float(u_mag.max()), u_mag_mean_m=float(u_mag.mean()),
    svm_max_Pa=float(svm_np.max()), svm_mean_Pa=float(svm_np.mean()),
    final_loss_eq =history["eq"][-1]  if history["eq"]  else None,
    final_loss_inn=history["inn"][-1] if history["inn"] else None,
    final_loss_out=history["out"][-1] if history["out"] else None,
)
json_file = f"metrics_{tag}.json"
with open(json_file,"w") as f: json.dump(metrics,f,indent=2)
print(f"JSON -> {json_file}")

# ============================================================
# 12. Optional ANSYS comparison
# ============================================================
#
# ANSYS Mechanical export:
#   Solution -> Export -> Nodal Solution Data
# Required columns:
#   x_mm, y_mm, z_mm, u_x_m, u_y_m, u_z_m,
#   u_mag_m, sigma_vm
#
#
# Set ANSYS_RESULTS_CSV at the top of the file.

if ANSYS_RESULTS_CSV and os.path.isfile(ANSYS_RESULTS_CSV):
    from scipy.spatial import cKDTree
    print(f"\n{'='*60}")
    print(f"ANSYS comparison: {ANSYS_RESULTS_CSV}")
    print(f"{'='*60}")
    ans  = np.genfromtxt(ANSYS_RESULTS_CSV, delimiter=",", names=True)
    axyz = np.column_stack([ans["x_mm"], ans["y_mm"], ans["z_mm"]])
    a_um = ans["u_mag_m"]
    a_vm = ans["sigma_vm"]
    _, idx = cKDTree(pts*1e3).query(axyz)
    p_um = u_mag[idx];  p_vm = svm_np[idx]
    def rel_L2(p, r):
        return np.sqrt(np.mean((p-r)**2))/(np.sqrt(np.mean(r**2))+1e-20)
    err_u  = rel_L2(p_um, a_um)
    err_vm = rel_L2(p_vm, a_vm)
    print(f"  Rel L2  |u|  : {err_u:.4e}")
    print(f"  Rel L2  s_vm : {err_vm:.4e}")
    np.savetxt(f"comparison_{tag}.csv",
               np.column_stack([axyz, p_um, a_um, np.abs(p_um-a_um),
                                 p_vm, a_vm, np.abs(p_vm-a_vm)]),
               delimiter=",",
               header="x_mm,y_mm,z_mm,u_pinn,u_ansys,u_err,svm_pinn,svm_ansys,svm_err",
               comments="")
    fig, axes = plt.subplots(1,3,figsize=(16,5))
    fig.suptitle("PINN vs ANSYS")
    axes[0].scatter(a_um*1e6, p_um*1e6, s=3, alpha=0.4)
    lim=[min(a_um.min(),p_um.min())*1e6, max(a_um.max(),p_um.max())*1e6]
    axes[0].plot(lim,lim,"r--"); axes[0].set_xlabel("ANSYS |u| (um)"); axes[0].set_ylabel("PINN |u| (um)")
    axes[0].set_title(f"|u| relL2={err_u:.2e}"); axes[0].grid(True,alpha=0.3)
    axes[1].scatter(a_vm/1e3, p_vm/1e3, s=3, alpha=0.4)
    lim=[min(a_vm.min(),p_vm.min())/1e3, max(a_vm.max(),p_vm.max())/1e3]
    axes[1].plot(lim,lim,"r--"); axes[1].set_xlabel("ANSYS s_vm (kPa)"); axes[1].set_ylabel("PINN s_vm (kPa)")
    axes[1].set_title(f"s_vm relL2={err_vm:.2e}"); axes[1].grid(True,alpha=0.3)
    axes[2].scatter(axyz[:,1], np.abs(p_um-a_um)*1e6, s=3, alpha=0.4)
    axes[2].set_xlabel("y (mm)"); axes[2].set_ylabel("|u_err| (um)"); axes[2].set_title("|u| error vs y")
    axes[2].grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(f"comparison_{tag}.png",dpi=150); plt.show()
    metrics["err_u_vs_ansys"]=float(err_u); metrics["err_vm_vs_ansys"]=float(err_vm)
    with open(json_file,"w") as f: json.dump(metrics,f,indent=2)
else:
    if ANSYS_RESULTS_CSV:
        print(f"\n[WARNING] ANSYS CSV not found: '{ANSYS_RESULTS_CSV}'")
    print("\nANSYS comparison skipped.")
    print("  Set ANSYS_RESULTS_CSV = '<your_file>.csv' when ready.")

# ============================================================
# 13. Plots
# ============================================================

# Loss history
fig, axes = plt.subplots(3,1,figsize=(10,9))
axes[0].semilogy(history["eq"]);   axes[0].set_title("Navier-Cauchy PDE residual")
axes[1].semilogy(history["inn"],label="inner"); axes[1].semilogy(history["out"],label="outer")
axes[1].set_title("Traction BC loss"); axes[1].legend()
axes[2].semilogy(history["z0"],label="y=0"); axes[2].semilogy(history["zL"],label="y=L")
axes[2].set_title("End-face Dirichlet penalty"); axes[2].legend()
axes[2].set_xlabel("Epoch")
for ax in axes: ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig("loss_history.png",dpi=150); plt.show()

# r-y scatter plots
y_n = pts[:,1]*1e3
r_n = np.sqrt(pts[:,0]**2+pts[:,2]**2)*1e3
fig,axes=plt.subplots(1,2,figsize=(14,5))
sc=axes[0].scatter(y_n,r_n,c=u_mag*1e6,cmap="viridis",s=2)
plt.colorbar(sc,ax=axes[0]).set_label("|u| (um)")
axes[0].set_xlabel("y (mm)"); axes[0].set_ylabel("r (mm)"); axes[0].set_title("Total displacement")
sc=axes[1].scatter(y_n,r_n,c=svm_np/1e3,cmap="plasma",s=2)
plt.colorbar(sc,ax=axes[1]).set_label("s_vm (kPa)")
axes[1].set_xlabel("y (mm)"); axes[1].set_ylabel("r (mm)"); axes[1].set_title("von Mises stress")
plt.tight_layout(); plt.savefig("results_ry_scatter.png",dpi=150); plt.show()

# Radial profiles
fig,axes=plt.subplots(1,2,figsize=(12,5)); fig.suptitle("Radial profiles at axial slices")
for label,y_sl in [("y=L/4",L/4),("y=L/2",L/2),("y=3L/4",3*L/4)]:
    mask=np.abs(pts[:,1]-y_sl)<0.005*L
    if mask.sum()<3: continue
    rs=r_n[mask]; idx_s=np.argsort(rs)
    axes[0].plot(rs[idx_s],u_mag[mask][idx_s]*1e6,"-o",ms=2,label=label)
    axes[1].plot(rs[idx_s],svm_np[mask][idx_s]/1e3,"-o",ms=2,label=label)
for ax,yl,ttl in zip(axes,["|u| (um)","s_vm (kPa)"],["Displacement","von Mises"]):
    ax.set_xlabel("r (mm)"); ax.set_ylabel(yl); ax.set_title(ttl)
    ax.legend(); ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig("radial_profiles.png",dpi=150); plt.show()

# 3D von Mises scatter
fig=plt.figure(figsize=(9,8)); ax3=fig.add_subplot(111,projection="3d")
sc3=ax3.scatter(pts[:,0]*1e3,pts[:,1]*1e3,pts[:,2]*1e3,
                c=svm_np/1e3,cmap="plasma",s=1,alpha=0.4)
plt.colorbar(sc3,ax=ax3).set_label("s_vm (kPa)")
ax3.set_xlabel("x (mm)"); ax3.set_ylabel("y (mm)"); ax3.set_zlabel("z (mm)")
ax3.set_title("von Mises — 3D")
plt.tight_layout(); plt.savefig("vm_stress_3d.png",dpi=150); plt.show()

# 3D displacement scatter
fig=plt.figure(figsize=(9,8)); ax3=fig.add_subplot(111,projection="3d")
sc3=ax3.scatter(pts[:,0]*1e3,pts[:,1]*1e3,pts[:,2]*1e3,
                c=u_mag*1e6,cmap="viridis",s=1,alpha=0.4)
plt.colorbar(sc3,ax=ax3).set_label("|u| (\u03bcm)")
ax3.set_xlabel("x (mm)"); ax3.set_ylabel("y (mm)"); ax3.set_zlabel("z (mm)")
ax3.set_title("Total displacement |u| — 3D")
plt.tight_layout(); plt.savefig("disp_3d.png",dpi=150); plt.show()

# Final summary
print(f"\n{'='*52}")
print(f"  FINAL SUMMARY")
print(f"{'='*52}")
print(f"  Max von Mises stress       = {svm_np.max():.3e} Pa")
print(f"  Max displacement magnitude = {u_mag.max():.3e} m")
print(f"{'='*52}")
print(f"\nDone. Weights -> model_{tag}.pt")
