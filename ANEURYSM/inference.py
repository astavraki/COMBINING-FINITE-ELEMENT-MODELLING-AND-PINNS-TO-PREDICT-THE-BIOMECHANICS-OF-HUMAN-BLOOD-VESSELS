"""
PINN inference — 3D aneurysm geometry
=====================================

Random inference points are generated from the Gmsh tetrahedral mesh and
used to evaluate displacement and stress fields.

Coordinate convention:
  x, z : cross-section plane
  y    : axial direction

Main outputs:
  - CSV file with displacement and stress values
  - MATLAB file for the outer-wall deformed shape
  - Point-cloud, plane-contour and y-r scatter plots
"""

import argparse
import csv
import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

try:
    import meshio
except ImportError:
    os.system("pip install meshio --quiet --break-system-packages")
    import meshio

try:
    from scipy.interpolate import griddata
    from scipy.io import savemat
except ImportError:
    os.system("pip install scipy --quiet --break-system-packages")
    from scipy.interpolate import griddata
    from scipy.io import savemat

# ============================================================
# Physical parameters
# ============================================================
MESH_SCALE = 1e-3          # Gmsh mesh is in mm, convert to metres
E = 5e5                    # Pa
nu = 0.45
p_i = 100 * 133.322        # Pa
p_o = 10  * 133.322        # Pa

mu = E / (2.0 * (1.0 + nu))
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

# Geometry-dependent values
L = None
r_max = None
u_scale = None


# ============================================================
# Network
# ============================================================
class AneurysmPINN(nn.Module):
    """PINN architecture used for the aneurysm case."""
    def __init__(self, n_hidden: int = 8, n_neurons: int = 100):
        super().__init__()
        layers = [nn.Linear(4, n_neurons), nn.Tanh()]
        for _ in range(n_hidden):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 3)]
        self.net = nn.Sequential(*layers)
        self.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global L, r_max, u_scale

        x_c = x[:, 0:1]
        y_c = x[:, 1:2]
        z_c = x[:, 2:3]

        r = torch.sqrt(x_c**2 + z_c**2) + 1e-12
        cos_t = x_c / r
        sin_t = z_c / r
        r_n = r / r_max
        y_n = 2.0 * y_c / L - 1.0

        feat = torch.cat([r_n, y_n, cos_t, sin_t], dim=1)
        out = self.net(feat)

        u_r_raw = out[:, 0:1] * u_scale
        u_y_raw = out[:, 1:2] * u_scale
        u_t_raw = out[:, 2:3] * u_scale

        # axial displacement constraint
        phi = y_c * (L - y_c) / (0.5 * L)**2

        u_x = u_r_raw * cos_t - u_t_raw * sin_t
        u_y = u_y_raw * phi
        u_z = u_r_raw * sin_t + u_t_raw * cos_t

        return torch.cat([u_x, u_y, u_z], dim=1)


# ============================================================
# Stress calculation
# ============================================================
def grad_vec(f, x, retain_graph=False):
    return torch.autograd.grad(
        f, x,
        grad_outputs=torch.ones_like(f),
        create_graph=False,
        retain_graph=retain_graph
    )[0]


def strains_stresses_infer(u, x):
    u1 = u[:, 0:1]
    u2 = u[:, 1:2]
    u3 = u[:, 2:3]

    g1 = grad_vec(u1, x, retain_graph=True)
    g2 = grad_vec(u2, x, retain_graph=True)
    g3 = grad_vec(u3, x, retain_graph=False)

    e11 = g1[:, 0:1]
    e22 = g2[:, 1:2]
    e33 = g3[:, 2:3]
    e12 = 0.5 * (g1[:, 1:2] + g2[:, 0:1])
    e13 = 0.5 * (g1[:, 2:3] + g3[:, 0:1])
    e23 = 0.5 * (g2[:, 2:3] + g3[:, 1:2])

    tr = e11 + e22 + e33

    s11 = lam * tr + 2.0 * mu * e11
    s22 = lam * tr + 2.0 * mu * e22
    s33 = lam * tr + 2.0 * mu * e33
    s12 = 2.0 * mu * e12
    s13 = 2.0 * mu * e13
    s23 = 2.0 * mu * e23

    s_vm = torch.sqrt(
        0.5 * (
            (s11 - s22)**2 +
            (s22 - s33)**2 +
            (s33 - s11)**2 +
            6.0 * (s12**2 + s13**2 + s23**2)
        )
    )

    return s11, s22, s33, s12, s13, s23, s_vm, e11, e22, e33


# ============================================================
# Mesh loading and sampling
# ============================================================
def load_mesh(mesh_file: str):
    mesh = meshio.read(mesh_file)
    pts = mesh.points * MESH_SCALE

    tets = None
    triangles = []
    for cb in mesh.cells:
        if cb.type == "tetra":
            tets = cb.data
        elif cb.type == "triangle":
            triangles.append(cb.data)

    if tets is None:
        raise ValueError("No tetrahedral cells found in the Gmsh mesh.")
    if not triangles:
        raise ValueError("No triangular surface cells found in the Gmsh mesh.")

    all_tris = np.vstack(triangles)
    return mesh, pts, tets, all_tris


def compute_surface_normals(pts, tets, all_tris):
    """Surface classification and outward normals."""
    tri_cents = pts[all_tris].mean(axis=1)
    L_local = pts[:, 1].max()
    tol_y = 1e-7

    is_end0 = tri_cents[:, 1] < tol_y
    is_endL = np.abs(tri_cents[:, 1] - L_local) < tol_y
    is_wall = ~(is_end0 | is_endL)

    local_faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    tet_face_map = {}
    for tet in tets:
        for lf in local_faces:
            key = frozenset(tet[list(lf)])
            opp = next(tet[j] for j in range(4) if j not in lf)
            tet_face_map[key] = opp

    wall_tris = all_tris[is_wall]
    wall_normals = np.zeros((len(wall_tris), 3), dtype=float)

    for k, tri in enumerate(wall_tris):
        opp = tet_face_map.get(frozenset(tri), None)
        if opp is None:
            continue
        v0, v1, v2 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        n = np.cross(v1 - v0, v2 - v0).astype(float)
        if np.dot(n, v0 - pts[opp]) < 0:
            n = -n
        wall_normals[k] = n / (np.linalg.norm(n) + 1e-15)

    wc = tri_cents[is_wall]
    r_wc = np.sqrt(wc[:, 0]**2 + wc[:, 2]**2) + 1e-15
    r_hat = np.column_stack([wc[:, 0] / r_wc, np.zeros(len(wc)), wc[:, 2] / r_wc])

    inner_wall_mask = (wall_normals * r_hat).sum(axis=1) < 0
    outer_wall_mask = ~inner_wall_mask

    data = {
        "tri_cents": tri_cents,
        "wall_tris": wall_tris,
        "wall_normals": wall_normals,
        "inn_tris": wall_tris[inner_wall_mask],
        "out_tris": wall_tris[outer_wall_mask],
        "inn_normals": wall_normals[inner_wall_mask],
        "out_normals": wall_normals[outer_wall_mask],
        "end0_tris": all_tris[is_end0],
        "endL_tris": all_tris[is_endL],
        "r_wc": r_wc,
        "inner_wall_mask": inner_wall_mask,
        "outer_wall_mask": outer_wall_mask,
    }
    return data


def sample_volume_random(pts, tets, n, seed=None):
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(tets), n)
    bary = g.dirichlet([1, 1, 1, 1], size=n)
    return (
        bary[:, 0:1] * pts[tets[idx, 0]] +
        bary[:, 1:2] * pts[tets[idx, 1]] +
        bary[:, 2:3] * pts[tets[idx, 2]] +
        bary[:, 3:4] * pts[tets[idx, 3]]
    )


def sample_surface_random(pts, tri_idx_arr, n, seed=None):
    g = np.random.default_rng(seed)
    chosen = g.integers(0, len(tri_idx_arr), n)
    u, v = g.random(n), g.random(n)
    bad = u + v > 1.0
    u[bad] = 1.0 - u[bad]
    v[bad] = 1.0 - v[bad]
    w = 1.0 - u - v
    tris = tri_idx_arr[chosen]
    return (
        w[:, None] * pts[tris[:, 0]] +
        u[:, None] * pts[tris[:, 1]] +
        v[:, None] * pts[tris[:, 2]]
    )


def compute_scales(pts, surface_data):
    """Geometry-based displacement scaling."""
    global L, r_max, u_scale

    L = float(pts[:, 1].max())
    r_max = float(np.sqrt(pts[:, 0]**2 + pts[:, 2]**2).max())

    r_wc = surface_data["r_wc"]
    inner_mask = surface_data["inner_wall_mask"]
    outer_mask = surface_data["outer_wall_mask"]

    r_i_est = float(r_wc[inner_mask].mean())
    r_o_est = float(r_wc[outer_mask].mean())
    t_est = r_o_est - r_i_est

    B_s = (p_i - p_o) / (1.0 / r_i_est**2 - 1.0 / r_o_est**2)
    A_s = -p_o + B_s / r_o_est**2
    u_r_est = lambda r: ((1.0 + nu) / E) * ((1.0 - 2.0 * nu) * A_s * r + B_s / r)

    u_scale_base = float(max(abs(u_r_est(r_i_est)), abs(u_r_est(r_o_est))))

    # displacement and PDE scaling
    u_scale = 1.5 * u_scale_base
    pde_scale = 0.5 * (p_i / t_est)

    return r_i_est, r_o_est, t_est, u_scale_base, u_scale, pde_scale


# ============================================================
# Inference
# ============================================================
def run_inference(model, pts_np, device, batch_size=4096):
    fields = {
        k: [] for k in [
            "u_x", "u_y", "u_z", "u_mag",
            "s11", "s22", "s33", "s12", "s13", "s23", "s_vm",
            "e11", "e22", "e33"
        ]
    }

    for start in range(0, len(pts_np), batch_size):
        end = min(start + batch_size, len(pts_np))
        x_t = torch.tensor(pts_np[start:end], dtype=torch.float32, device=device, requires_grad=True)

        with torch.enable_grad():
            u = model(x_t)
            s11, s22, s33, s12, s13, s23, svm, e11, e22, e33 = strains_stresses_infer(u, x_t)

        def g(t):
            return t.detach().cpu().numpy().ravel()

        fields["u_x"].append(g(u[:, 0:1]))
        fields["u_y"].append(g(u[:, 1:2]))
        fields["u_z"].append(g(u[:, 2:3]))
        fields["u_mag"].append(g(torch.sqrt((u**2).sum(dim=1, keepdim=True))))
        fields["s11"].append(g(s11))
        fields["s22"].append(g(s22))
        fields["s33"].append(g(s33))
        fields["s12"].append(g(s12))
        fields["s13"].append(g(s13))
        fields["s23"].append(g(s23))
        fields["s_vm"].append(g(svm))
        fields["e11"].append(g(e11))
        fields["e22"].append(g(e22))
        fields["e33"].append(g(e33))

        print(f"  processed {end}/{len(pts_np)}")

    return {k: np.concatenate(v) for k, v in fields.items()}


def write_csv(path, pts_np, fields, labels):
    header = [
        "x_m", "y_m", "z_m", "r_m", "point_type",
        "u_x_m", "u_y_m", "u_z_m", "u_mag_m",
        "sigma_xx_Pa", "sigma_yy_Pa", "sigma_zz_Pa",
        "sigma_xy_Pa", "sigma_xz_Pa", "sigma_yz_Pa", "sigma_vm_Pa",
        "e11", "e22", "e33"
    ]
    r = np.sqrt(pts_np[:, 0]**2 + pts_np[:, 2]**2)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(pts_np)):
            w.writerow([
                pts_np[i, 0], pts_np[i, 1], pts_np[i, 2], r[i], labels[i],
                fields["u_x"][i], fields["u_y"][i], fields["u_z"][i], fields["u_mag"][i],
                fields["s11"][i], fields["s22"][i], fields["s33"][i],
                fields["s12"][i], fields["s13"][i], fields["s23"][i], fields["s_vm"][i],
                fields["e11"][i], fields["e22"][i], fields["e33"][i]
            ])
    print(f"Saved CSV -> {path}")


# ============================================================
# Plots
# ============================================================
def plot_random_3d(pts_np, values, title, cbar_label, filename, cmap="viridis"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        pts_np[:, 0] * 1e3,
        pts_np[:, 1] * 1e3,
        pts_np[:, 2] * 1e3,
        c=values,
        cmap=cmap,
        s=2,
        alpha=0.45,
        linewidths=0
    )
    plt.colorbar(sc, ax=ax, pad=0.12, shrink=0.6).set_label(cbar_label)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved -> {filename}")


def plot_ry_scatter(pts_np, fields):
    y_mm = pts_np[:, 1] * 1e3
    r_mm = np.sqrt(pts_np[:, 0]**2 + pts_np[:, 2]**2) * 1e3

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sc = axes[0].scatter(y_mm, r_mm, c=fields["u_mag"] * 1e6, cmap="viridis", s=2, alpha=0.5)
    plt.colorbar(sc, ax=axes[0]).set_label("|u| (um)")
    axes[0].set_xlabel("y (mm)")
    axes[0].set_ylabel("r (mm)")
    axes[0].set_title("Random inference points: total deformation")

    sc = axes[1].scatter(y_mm, r_mm, c=fields["s_vm"] / 1e3, cmap="plasma", s=2, alpha=0.5)
    plt.colorbar(sc, ax=axes[1]).set_label("sigma_vm (kPa)")
    axes[1].set_xlabel("y (mm)")
    axes[1].set_ylabel("r (mm)")
    axes[1].set_title("Random inference points: von Mises stress")

    plt.tight_layout()
    plt.savefig("inference_random_ry_scatter.png", dpi=150)
    plt.show()
    print("Saved -> inference_random_ry_scatter.png")


def plot_plane_contours(pts_np, fields, plane_tol_factor=0.015):
    plane_defs = [("L/3", L / 3.0), ("L/2", L / 2.0), ("3L/4", 3.0 * L / 4.0)]
    plane_tol = plane_tol_factor * L

    print(f"\n{'='*60}")
    print("  PLANE SUMMARY  (PINN random inference points)")
    print(f"{'='*60}")

    plane_data = []
    for label, y_plane in plane_defs:
        mask = np.abs(pts_np[:, 1] - y_plane) < plane_tol

        if mask.sum() < 10:
            print(f"\n  At y={label} = {y_plane*1e3:.3f} mm: too few points in slice ({mask.sum()})")
            plane_data.append(None)
            continue

        pts_plane = pts_np[mask]
        umag_plane = fields["u_mag"][mask]
        svm_plane = fields["s_vm"][mask]

        print(f"\n  At y={label} = {y_plane*1e3:.3f} mm:")
        print(f"    random points in slice = {mask.sum()}")
        print(f"    Max |u|                = {umag_plane.max():.3e} m")
        print(f"    Max sigma_vm           = {svm_plane.max():.3e} Pa")

        plane_data.append((label, y_plane, pts_plane, umag_plane, svm_plane))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "PINN random-point inference at y = L/3, L/2, 3L/4\n"
        "Row 1: total deformation |u|     Row 2: von Mises stress sigma_vm",
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

        xi = np.linspace(x_mm.min(), x_mm.max(), 300)
        zi = np.linspace(z_mm.min(), z_mm.max(), 300)
        XI, ZI = np.meshgrid(xi, zi)

        U_grid = griddata((x_mm, z_mm), umag_plane, (XI, ZI), method="linear")
        S_grid = griddata((x_mm, z_mm), svm_plane, (XI, ZI), method="linear")

        valid = griddata((x_mm, z_mm), np.ones_like(x_mm), (XI, ZI), method="linear")
        U_grid = np.where(np.isnan(valid), np.nan, U_grid)
        S_grid = np.where(np.isnan(valid), np.nan, S_grid)

        R = np.sqrt(XI**2 + ZI**2)
        r_pts = np.sqrt(x_mm**2 + z_mm**2)
        r_min = r_pts.min()
        r_max_slice = r_pts.max()
        mask_ring = (R >= r_min) & (R <= r_max_slice)
        U_grid[~mask_ring] = np.nan
        S_grid[~mask_ring] = np.nan

        cf1 = axes[0, col].contourf(XI, ZI, U_grid, levels=40, cmap="viridis")
        axes[0, col].scatter(x_mm, z_mm, s=1, c="k", alpha=0.06)
        plt.colorbar(cf1, ax=axes[0, col]).set_label("|u| (m)")
        axes[0, col].set_aspect("equal")
        axes[0, col].set_xlabel("x (mm)")
        axes[0, col].set_ylabel("z (mm)")
        axes[0, col].set_title(f"|u| at y={label} = {y_plane*1e3:.2f} mm")

        cf2 = axes[1, col].contourf(XI, ZI, S_grid, levels=40, cmap="plasma")
        axes[1, col].scatter(x_mm, z_mm, s=1, c="k", alpha=0.06)
        plt.colorbar(cf2, ax=axes[1, col]).set_label("sigma_vm (Pa)")
        axes[1, col].set_aspect("equal")
        axes[1, col].set_xlabel("x (mm)")
        axes[1, col].set_ylabel("z (mm)")
        axes[1, col].set_title(f"sigma_vm at y={label} = {y_plane*1e3:.2f} mm")

    plt.tight_layout()
    plt.savefig("inference_random_plane_contours.png", dpi=150)
    plt.show()
    print("Saved -> inference_random_plane_contours.png")



def export_matlab_outer_surface(model, pts, surface_data, device, filename="pinn_inference_results.mat",
                                n_theta=96, n_y=80):
    """Export a structured outer-surface dataset for MATLAB."""
    global L

    out_tris = surface_data["out_tris"]
    if len(out_tris) == 0:
        raise ValueError("No outer-wall triangles found, cannot export MATLAB surface.")

    wc_out = pts[out_tris].mean(axis=1)
    y_out = wc_out[:, 1]
    r_out = np.sqrt(wc_out[:, 0]**2 + wc_out[:, 2]**2)

    y_vec = np.linspace(0.0, L, n_y)
    theta_vec = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    # outer radius profile from the Gmsh surface
    n_bins = max(60, n_y)
    bin_edges = np.linspace(0.0, L, n_bins + 1)
    y_bin = []
    r_bin = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        m = (y_out >= lo) & (y_out <= hi)
        if np.any(m):
            y_bin.append(0.5 * (lo + hi))
            # stable outer-boundary estimate
            r_bin.append(np.percentile(r_out[m], 95))

    y_bin = np.asarray(y_bin)
    r_bin = np.asarray(r_bin)

    if len(y_bin) < 2:
        # fallback radius profile
        r_profile = np.full_like(y_vec, r_out.max())
    else:
        r_profile = np.interp(y_vec, y_bin, r_bin, left=r_bin[0], right=r_bin[-1])

    TH, YY = np.meshgrid(theta_vec, y_vec, indexing="ij")
    RR = r_profile[None, :]

    X_outer = RR * np.cos(TH)
    Y_outer = YY
    Z_outer = RR * np.sin(TH)

    pts_eval = np.column_stack([
        X_outer.ravel(),
        Y_outer.ravel(),
        Z_outer.ravel()
    ])

    u_all = []
    batch = 8192
    model.eval()

    with torch.no_grad():
        for start in range(0, len(pts_eval), batch):
            end = min(start + batch, len(pts_eval))
            x_t = torch.tensor(pts_eval[start:end], dtype=torch.float32, device=device)
            u_t = model(x_t)
            u_all.append(u_t.detach().cpu().numpy())

    u_np = np.vstack(u_all)

    Ux_outer = u_np[:, 0].reshape(n_theta, n_y)
    Uy_outer = u_np[:, 1].reshape(n_theta, n_y)
    Uz_outer = u_np[:, 2].reshape(n_theta, n_y)
    Umag_outer = np.linalg.norm(u_np, axis=1).reshape(n_theta, n_y)

    savemat(filename, {
        "X_outer": X_outer,
        "Y_outer": Y_outer,
        "Z_outer": Z_outer,
        "u1_outer": Ux_outer,
        "u2_outer": Uy_outer,
        "u3_outer": Uz_outer,
        "umag_outer": Umag_outer,
        "theta_vec": theta_vec,
        "y_vec": y_vec,
        "r_outer_profile": r_profile,
        "L": L,
        "E": E,
        "nu": nu,
        "p_i": p_i,
        "p_o": p_o,
    })

    print(f"Saved MATLAB file -> {filename}")
    print("  Contains: X_outer, Y_outer, Z_outer, u1_outer, u2_outer, u3_outer, umag_outer")


# ============================================================
# Main execution
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default="Part1.msh")
    parser.add_argument("--model", default="model_aneurysm_Ni4000_Nw1200_ep6000.pt")
    parser.add_argument("--out", default="aneurysm_random_inference.csv")
    parser.add_argument("--mat", default="pinn_inference_results.mat", help="MATLAB output for outer deformed shape")
    parser.add_argument("--n-volume", type=int, default=80000, help="random volume inference points")
    parser.add_argument("--n-inner", type=int, default=8000, help="random inner wall inference points")
    parser.add_argument("--n-outer", type=int, default=8000, help="random outer wall inference points")
    parser.add_argument("--n-end", type=int, default=2000, help="random points per end face")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--neurons", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mesh, pts, tets, all_tris = load_mesh(args.mesh)
    surface_data = compute_surface_normals(pts, tets, all_tris)
    r_i_est, r_o_est, t_est, u_scale_base, u_scale_final, pde_scale = compute_scales(pts, surface_data)

    print("\nGeometry and scale:")
    print(f"  nodes        = {len(pts)}")
    print(f"  tetrahedra   = {len(tets)}")
    print(f"  L            = {L*1e3:.3f} mm")
    print(f"  r_max        = {r_max*1e3:.3f} mm")
    print(f"  r_i_est      = {r_i_est*1e3:.3f} mm")
    print(f"  r_o_est      = {r_o_est*1e3:.3f} mm")
    print(f"  t_est        = {t_est*1e3:.3f} mm")
    print(f"  u_scale base = {u_scale_base:.3e} m")
    print(f"  u_scale used = {u_scale_final:.3e} m")
    print(f"  pde_scale    = {pde_scale:.3e} Pa/m")

    print("\nSurface classification:")
    print(f"  inner wall triangles = {len(surface_data['inn_tris'])}")
    print(f"  outer wall triangles = {len(surface_data['out_tris'])}")
    print(f"  y=0 end triangles    = {len(surface_data['end0_tris'])}")
    print(f"  y=L end triangles    = {len(surface_data['endL_tris'])}")

    # Random inference points
    print("\nGenerating random inference points from the Gmsh geometry ...")
    vol_pts = sample_volume_random(pts, tets, args.n_volume, seed=args.seed)
    inn_pts = sample_surface_random(pts, surface_data["inn_tris"], args.n_inner, seed=args.seed + 1)
    out_pts = sample_surface_random(pts, surface_data["out_tris"], args.n_outer, seed=args.seed + 2)
    end0_pts = sample_surface_random(pts, surface_data["end0_tris"], args.n_end, seed=args.seed + 3)
    endL_pts = sample_surface_random(pts, surface_data["endL_tris"], args.n_end, seed=args.seed + 4)

    infer_pts = np.vstack([vol_pts, inn_pts, out_pts, end0_pts, endL_pts])
    labels = (
        ["volume_random"] * len(vol_pts) +
        ["inner_wall_random"] * len(inn_pts) +
        ["outer_wall_random"] * len(out_pts) +
        ["end_y0_random"] * len(end0_pts) +
        ["end_yL_random"] * len(endL_pts)
    )
    labels = np.array(labels, dtype=object)

    print(f"  volume points = {len(vol_pts)}")
    print(f"  inner wall    = {len(inn_pts)}")
    print(f"  outer wall    = {len(out_pts)}")
    print(f"  end y=0       = {len(end0_pts)}")
    print(f"  end y=L       = {len(endL_pts)}")
    print(f"  total         = {len(infer_pts)}")

    # Load model
    model = AneurysmPINN(n_hidden=args.hidden, n_neurons=args.neurons).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"\nLoaded model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    # Run inference
    print("\nRunning random-point inference ...")
    fields = run_inference(model, infer_pts, device, batch_size=args.batch)

    print("\nInference ranges:")
    print(f"  |u|      : [{fields['u_mag'].min():.3e}, {fields['u_mag'].max():.3e}] m")
    print(f"  sigma_vm : [{fields['s_vm'].min():.3e}, {fields['s_vm'].max():.3e}] Pa")

    write_csv(args.out, infer_pts, fields, labels)

    export_matlab_outer_surface(model, pts, surface_data, device, filename=args.mat)

    # Plots
    plot_random_3d(
        infer_pts,
        fields["u_mag"] * 1e6,
        "Aneurysm PINN random inference points: total deformation |u|",
        "|u| (um)",
        "inference_random_3d_disp.png",
        cmap="viridis"
    )

    plot_random_3d(
        infer_pts,
        fields["s_vm"] / 1e3,
        "Aneurysm PINN random inference points: von Mises stress",
        "sigma_vm (kPa)",
        "inference_random_3d_svm.png",
        cmap="plasma"
    )

    plot_ry_scatter(infer_pts, fields)
    plot_plane_contours(infer_pts, fields)

    print(f"\n{'='*60}")
    print("  FINAL SUMMARY — RANDOM INFERENCE POINTS")
    print(f"{'='*60}")
    print(f"  Max |u|      = {fields['u_mag'].max():.3e} m")
    print(f"  Max sigma_vm = {fields['s_vm'].max():.3e} Pa")
    print(f"  CSV          = {args.out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
