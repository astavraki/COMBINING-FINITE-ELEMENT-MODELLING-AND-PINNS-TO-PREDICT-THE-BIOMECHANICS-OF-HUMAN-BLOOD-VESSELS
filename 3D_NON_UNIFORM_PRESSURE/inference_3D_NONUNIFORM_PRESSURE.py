"""PINN inference — 3D pipe with non-uniform internal pressure

The trained model is evaluated on the Gmsh mesh nodes and tetrahedral
element centres. The script exports displacement and stress results for
post-processing in MATLAB, ParaView, or Gmsh.

The internal pressure varies along the pipe length and reaches its maximum
at the midplane. Both ends are fixed through the displacement constraint used
in the network architecture."""

import argparse
import math
import csv
import numpy as np
import torch
import torch.nn as nn
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
r_o     = r_i + t
mu      = E / (2.0 * (1.0 + nu))
lam     = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

u_scale = 3.0e-3
print(f"u_scale = {u_scale:.4e} m")


# Internal pressure distribution
def p_i_of_z(z):
    z = np.asarray(z, dtype=float)
    return p_i_mid * 0.5 * (1.0 - np.cos(2.0 * np.pi * z / L))


# ============================================================
# 2. Network
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
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x3 = x[:, 2:3]

        r = torch.sqrt(x1**2 + x2**2) + 1e-12
        cos_t = x1 / r
        sin_t = x2 / r

        feat = torch.cat([
            2 * (r - r_i) / (r_o - r_i) - 1,
            cos_t,
            sin_t,
            2 * x3 / L - 1
        ], dim=1)

        out = self.net(feat)
        u_r   = out[:, 0:1] * u_scale
        u_th  = out[:, 1:2] * u_scale
        u_z_r = out[:, 2:3] * u_scale

        phi = x3 * (L - x3) / (0.5 * L)**2

        u1 = (u_r * cos_t - u_th * sin_t) * phi
        u2 = (u_r * sin_t + u_th * cos_t) * phi
        u3 = u_z_r * phi
        return torch.cat([u1, u2, u3], dim=1)


# ============================================================
# 3. Stress calculation
# ============================================================
def grad_vec(f, x, retain_graph=False):
    return torch.autograd.grad(
        f, x,
        grad_outputs=torch.ones_like(f),
        create_graph=False,
        retain_graph=retain_graph
    )[0]


def strains_stresses(u, x):
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

    s11 = lam * tr + 2 * mu * e11
    s22 = lam * tr + 2 * mu * e22
    s33 = lam * tr + 2 * mu * e33
    s12 = 2 * mu * e12
    s13 = 2 * mu * e13
    s23 = 2 * mu * e23

    r = torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2) + 1e-12
    c = x[:, 0:1] / r
    s_ = x[:, 1:2] / r

    s_rr = s11 * c**2 + 2 * s12 * c * s_ + s22 * s_**2
    s_tt = s11 * s_**2 - 2 * s12 * c * s_ + s22 * c**2

    s_vm = torch.sqrt(
        0.5 * (
            (s11 - s22)**2 +
            (s22 - s33)**2 +
            (s33 - s11)**2 +
            6 * (s12**2 + s13**2 + s23**2)
        )
    )

    return s11, s22, s33, s12, s13, s23, s_rr, s_tt, s_vm, e11, e22, e33


# ============================================================
# 4. Gmsh mesh reader
# ============================================================
def read_gmsh4(path):
    with open(path, 'r') as f:
        lines = [l.rstrip('\r\n') for l in f]

    nd_start = lines.index('$Nodes') + 1
    nd_end   = lines.index('$EndNodes')

    node_coords = {}
    i = nd_start + 1
    while i < nd_end:
        parts = lines[i].split()
        nblock = int(parts[3])
        tags = [int(lines[i + j + 1].strip()) for j in range(nblock)]
        cs = i + nblock + 1
        for j, tag in enumerate(tags):
            node_coords[tag] = list(map(float, lines[cs + j].split()))
        i = cs + nblock

    tag2idx = {}
    cl = []
    for tag in sorted(node_coords):
        tag2idx[tag] = len(cl)
        cl.append(node_coords[tag])

    nodes_mm = np.array(cl, dtype=np.float64)

    el_start = lines.index('$Elements') + 1
    el_end   = lines.index('$EndElements')
    tet_conn = []
    i = el_start + 1
    while i < el_end:
        parts = lines[i].split()
        etype, nelem = int(parts[2]), int(parts[3])
        if etype == 4:
            for j in range(1, nelem + 1):
                row = lines[i + j].split()
                tet_conn.append([tag2idx[int(row[k])] for k in range(1, 5)])
        i += nelem + 1

    return nodes_mm * 1e-3, np.array(tet_conn, dtype=np.int64)


# ============================================================
# 5. Batched inference
# ============================================================
def run_inference(model, pts_np, device, batch_size=4096):
    N = len(pts_np)
    fields = {
        k: [] for k in [
            'u1', 'u2', 'u3', 'umag', 's11', 's22', 's33',
            's12', 's13', 's23', 's_rr', 's_tt', 's_vm', 'e11', 'e22', 'e33'
        ]
    }

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_t = torch.tensor(
            pts_np[start:end],
            dtype=torch.float64,
            device=device,
            requires_grad=True
        )
        with torch.enable_grad():
            u = model(x_t)
            s11, s22, s33, s12, s13, s23, srr, stt, svm, e11, e22, e33 = strains_stresses(u, x_t)

        def g(t):
            return t.detach().cpu().numpy().ravel()

        fields['u1'].append(g(u[:, 0:1]))
        fields['u2'].append(g(u[:, 1:2]))
        fields['u3'].append(g(u[:, 2:3]))
        fields['umag'].append(g(torch.sqrt((u**2).sum(dim=1, keepdim=True))))
        fields['s11'].append(g(s11))
        fields['s22'].append(g(s22))
        fields['s33'].append(g(s33))
        fields['s12'].append(g(s12))
        fields['s13'].append(g(s13))
        fields['s23'].append(g(s23))
        fields['s_rr'].append(g(srr))
        fields['s_tt'].append(g(stt))
        fields['s_vm'].append(g(svm))
        fields['e11'].append(g(e11))
        fields['e22'].append(g(e22))
        fields['e33'].append(g(e33))

        if (start // batch_size) % 10 == 0:
            print(f"  processed {end}/{N} …")

    return {k: np.concatenate(v) for k, v in fields.items()}


# ============================================================
# 6. Main routine
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh',  default='circleFace.msh')
    parser.add_argument('--model', default='model_Ni1344_Nw192_ep8000.pt')
    parser.add_argument('--out',   default='inference_results.csv')
    parser.add_argument('--batch', type=int, default=4096)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    # Load trained model
    model = PipePINN(n_hidden=8, n_neurons=100).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Loaded model — {sum(p.numel() for p in model.parameters())} params")

    # Read mesh and element centres
    nodes, tet_conn = read_gmsh4(args.mesh)
    tet_centres = nodes[tet_conn].mean(axis=1)
    print(f"Nodes: {len(nodes)}  Tets: {len(tet_conn)}")

    def in_domain(pts):
        r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        return (
            (r >= r_i * 0.999) &
            (r <= r_o * 1.001) &
            (pts[:, 2] >= -1e-6) &
            (pts[:, 2] <= L + 1e-6)
        )

    nodes_ok   = nodes[in_domain(nodes)]
    centres_ok = tet_centres[in_domain(tet_centres)]
    print(f"In domain — nodes: {len(nodes_ok)}  centres: {len(centres_ok)}")

    # Mesh point categories
    tol_wall = (r_o - r_i) * 0.05
    tol_end  = L * 0.02
    node_r = np.sqrt(nodes_ok[:, 0]**2 + nodes_ok[:, 1]**2)

    mask_inn = node_r < r_i + tol_wall
    mask_out = node_r > r_o - tol_wall
    mask_z0  = (~mask_inn) & (~mask_out) & (nodes_ok[:, 2] < tol_end)
    mask_zL  = (~mask_inn) & (~mask_out) & (nodes_ok[:, 2] > L - tol_end)
    mask_vol = (~mask_inn) & (~mask_out) & (~mask_z0) & (~mask_zL)

    # 3D point cloud and midplane view
    theta_c = np.linspace(0, 2 * math.pi, 300)

    def add_walls_2d(ax):
        for rv in [r_i, r_o]:
            ax.plot(rv * np.cos(theta_c) * 1e3, rv * np.sin(theta_c) * 1e3, 'k', lw=1.0, alpha=0.6)

    all_pts_2d = np.vstack([nodes_ok, centres_ok])
    all_cols_2d = (
        ['red'] * mask_inn.sum() +
        ['orange'] * mask_out.sum() +
        ['green'] * mask_z0.sum() +
        ['purple'] * mask_zL.sum() +
        ['grey'] * mask_vol.sum() +
        ['steelblue'] * len(centres_ok)
    )
    mid_sel = np.abs(all_pts_2d[:, 2] - 0.5 * L) < L * 0.06

    fig = plt.figure(figsize=(13, 7))
    ax3d = fig.add_subplot(121, projection='3d')

    def sc3(pts, c, lbl, s):
        if pts.shape[0] == 0:
            return
        ax3d.scatter(pts[:, 0] * 1e3, pts[:, 1] * 1e3, pts[:, 2] * 1e3,
                     s=s, c=c, label=lbl, alpha=0.35, linewidths=0)

    sc3(centres_ok, 'steelblue', f'Element centres ({len(centres_ok)})', 2)
    sc3(nodes_ok[mask_inn], 'red', f'Inner wall ({mask_inn.sum()})', 8)
    sc3(nodes_ok[mask_out], 'orange', f'Outer wall ({mask_out.sum()})', 8)
    sc3(nodes_ok[mask_z0], 'green', f'z=0 fixed ({mask_z0.sum()})', 8)
    sc3(nodes_ok[mask_zL], 'purple', f'z=L fixed ({mask_zL.sum()})', 8)
    sc3(nodes_ok[mask_vol], 'grey', f'Interior ({mask_vol.sum()})', 4)

    ax3d.set_xlabel('x1 (mm)')
    ax3d.set_ylabel('x2 (mm)')
    ax3d.set_zlabel('z (mm)')
    ax3d.set_title('Gmsh point cloud')
    ax3d.legend(markerscale=3, fontsize=8, loc='upper left')

    ax_xy = fig.add_subplot(122)
    ax_xy.scatter(
        all_pts_2d[mid_sel, 0] * 1e3,
        all_pts_2d[mid_sel, 1] * 1e3,
        c=np.array(all_cols_2d)[mid_sel],
        s=5, alpha=0.5, linewidths=0
    )
    add_walls_2d(ax_xy)
    ax_xy.set_aspect('equal')
    ax_xy.set_xlabel('x1 (mm)')
    ax_xy.set_ylabel('x2 (mm)')
    ax_xy.set_title(f'Cross-section z≈L/2 ({mid_sel.sum()} pts)')
    ax_xy.grid(True, alpha=0.2)

    fig.suptitle(
        f'{len(nodes_ok)} nodes + {len(centres_ok)} centres  |  z=0 fixed, z=L fixed',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('inference_collocation_points.png', dpi=150)
    plt.show()
    print('Saved → inference_collocation_points.png')

    # Run inference
    print("\n── Inference on mesh nodes ──────────────────")
    node_fields = run_inference(model, nodes_ok, device, args.batch)

    print("\n── Inference on element centres ─────────────")
    centre_fields = run_inference(model, centres_ok, device, args.batch)

    # CSV export
    def write_csv(path, pts, fields, label):
        header = [
            'x_m', 'y_m', 'z_m', 'r_m', 'point_type',
            'u1_m', 'u2_m', 'u3_m', 'umag_m',
            's11_Pa', 's22_Pa', 's33_Pa', 's12_Pa', 's13_Pa', 's23_Pa',
            's_rr_Pa', 's_tt_Pa', 's_vm_Pa', 'e11', 'e22', 'e33'
        ]
        r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(len(pts)):
                w.writerow([
                    pts[i, 0], pts[i, 1], pts[i, 2], r[i], label,
                    fields['u1'][i], fields['u2'][i], fields['u3'][i], fields['umag'][i],
                    fields['s11'][i], fields['s22'][i], fields['s33'][i],
                    fields['s12'][i], fields['s13'][i], fields['s23'][i],
                    fields['s_rr'][i], fields['s_tt'][i], fields['s_vm'][i],
                    fields['e11'][i], fields['e22'][i], fields['e33'][i]
                ])
        print(f"  Saved {len(pts)} rows → {path}")

    node_csv = args.out.replace('.csv', '_nodes.csv')
    cent_csv = args.out.replace('.csv', '_centres.csv')

    write_csv(node_csv, nodes_ok, node_fields, 'node')
    write_csv(cent_csv, centres_ok, centre_fields, 'element_centre')

    total_rows = 0
    with open(args.out, 'w', newline='') as fout:
        for idx, src in enumerate([node_csv, cent_csv]):
            with open(src, 'r', newline='') as fin:
                for lineno, line in enumerate(fin):
                    if lineno == 0 and idx > 0:
                        continue
                    fout.write(line)
                    if lineno > 0:
                        total_rows += 1
    print(f"  Combined → {args.out}  ({total_rows} rows)")

    # Regular cylindrical grid for contour plots
    Nq, Nr_q = 60, 40
    r_q = np.linspace(r_i, r_o, Nr_q)
    t_q = np.linspace(0, 2 * math.pi, Nq, endpoint=False)
    R_q, T_q = np.meshgrid(r_q, t_q, indexing='ij')
    t_cl = np.append(t_q, t_q[0] + 2 * math.pi)
    R_qc, T_qc = np.meshgrid(r_q, t_cl, indexing='ij')
    Xm = R_qc * np.cos(T_qc)
    Ym = R_qc * np.sin(T_qc)
    cc = lambda a: np.concatenate([a, a[:, :1]], axis=1)
    theta_wall = np.linspace(0, 2 * math.pi, 400)

    def add_walls(ax):
        for rv in [r_i, r_o]:
            ax.plot(rv * np.cos(theta_wall), rv * np.sin(theta_wall), 'k', lw=1.2)

    def eval_grid(z_val):
        xq = np.stack([
            (R_q * np.cos(T_q)).ravel(),
            (R_q * np.sin(T_q)).ravel(),
            np.full(Nr_q * Nq, z_val)
        ], axis=1)
        x_t = torch.tensor(xq, dtype=torch.float64, device=device, requires_grad=True)
        with torch.enable_grad():
            u_g = model(x_t)
            s11g, s22g, s33g, s12g, s13g, s23g, srrg, sttg, svmg, _, _, _ = strains_stresses(u_g, x_t)

        def g(t):
            return t.detach().cpu().numpy().ravel().reshape(Nr_q, Nq)

        return dict(
            umag=np.sqrt((u_g.detach().cpu().numpy()**2).sum(1)).reshape(Nr_q, Nq),
            u1=u_g[:, 0].detach().cpu().numpy().reshape(Nr_q, Nq),
            u2=u_g[:, 1].detach().cpu().numpy().reshape(Nr_q, Nq),
            u3=u_g[:, 2].detach().cpu().numpy().reshape(Nr_q, Nq),
            svm=g(svmg), srr=g(srrg), stt=g(sttg),
            s33=g(s33g), s13=g(s13g), s23=g(s23g)
        )

    # Per-plane contours
    detail_planes = [L/3, L/2, 3*L/4]
    detail_labels = ["L/3", "L/2", "3L/4"]
    detail_data = [eval_grid(zv) for zv in detail_planes]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Inference — PINN results at z = L/3,  L/2,  3L/4\n"
        "(both ends fixed  |  Row 1: |u|    Row 2: σ_vm)",
        fontsize=13
    )

    for col, (zv, zlbl, d) in enumerate(zip(detail_planes, detail_labels, detail_data)):
        ax_u = axes[0, col]
        cf_u = ax_u.contourf(Xm, Ym, cc(d['umag']), levels=30, cmap='viridis')
        plt.colorbar(cf_u, ax=ax_u).set_label("|u| (m)")
        add_walls(ax_u)
        ax_u.set_aspect('equal')
        ax_u.set_title(f"PINN |u|   z = {zlbl} = {zv*1e3:.2f} mm\np_i = {p_i_of_z(zv):.1f} Pa")
        ax_u.set_xlabel("x1 (m)")
        ax_u.set_ylabel("x2 (m)")

        ax_s = axes[1, col]
        cf_s = ax_s.contourf(Xm, Ym, cc(d['svm']), levels=30, cmap='plasma')
        plt.colorbar(cf_s, ax=ax_s).set_label("σ_vm (Pa)")
        add_walls(ax_s)
        ax_s.set_aspect('equal')
        ax_s.set_title(f"PINN σ_vm  z = {zlbl} = {zv*1e3:.2f} mm")
        ax_s.set_xlabel("x1 (m)")
        ax_s.set_ylabel("x2 (m)")

    plt.tight_layout()
    plt.savefig("inference_per_plane_detail.png", dpi=150)
    plt.show()
    print("Saved → inference_per_plane_detail.png")

    # Midplane contours
    mid = eval_grid(0.5 * L)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Inference — PINN results at z=L/2")

    cf = axes[0].contourf(Xm, Ym, cc(mid['umag']), levels=30, cmap='viridis')
    plt.colorbar(cf, ax=axes[0]).set_label("|u| (m)")
    add_walls(axes[0])
    axes[0].set_aspect('equal')
    axes[0].set_title("PINN |u|")
    axes[0].set_xlabel("x1 (m)")
    axes[0].set_ylabel("x2 (m)")

    cf = axes[1].contourf(Xm, Ym, cc(mid['svm']), levels=30, cmap='plasma')
    plt.colorbar(cf, ax=axes[1]).set_label("σ_vm (Pa)")
    add_walls(axes[1])
    axes[1].set_aspect('equal')
    axes[1].set_title("PINN σ_vm")
    axes[1].set_xlabel("x1 (m)")
    axes[1].set_ylabel("x2 (m)")

    plt.tight_layout()
    plt.savefig("inference_midplane_detail.png", dpi=150)
    plt.show()
    print("Saved → inference_midplane_detail.png")



    # Longitudinal cross-section
    Nx_half = 60
    Nz_long = 120

    x_left  = np.linspace(-r_o, -r_i, Nx_half)
    x_right = np.linspace( r_i,  r_o, Nx_half)
    x_full  = np.concatenate([x_left, x_right])
    z_vals  = np.linspace(0, L, Nz_long)

    X_long, Z_long = np.meshgrid(x_full, z_vals, indexing='ij')
    Y_long = np.zeros_like(X_long)

    pts_long = np.stack([X_long.ravel(), Y_long.ravel(), Z_long.ravel()], axis=1)
    x_t = torch.tensor(pts_long, dtype=torch.float64, device=device, requires_grad=True)

    with torch.enable_grad():
        u_long = model(x_t)
        _, _, _, _, _, _, _, _, svml, _, _, _ = strains_stresses(u_long, x_t)

    umag_long = torch.sqrt((u_long**2).sum(dim=1)).detach().cpu().numpy().reshape(len(x_full), Nz_long)
    svm_long  = svml.detach().cpu().numpy().reshape(len(x_full), Nz_long)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    cf1 = axes[0].contourf(Z_long * 1e3, X_long * 1e3, umag_long, levels=30, cmap='viridis')
    plt.colorbar(cf1, ax=axes[0]).set_label("|u| (m)")
    axes[0].set_title("Longitudinal full cross-section: total deformation |u|")
    axes[0].set_xlabel("z (mm)")
    axes[0].set_ylabel("x (mm)")

    cf2 = axes[1].contourf(Z_long * 1e3, X_long * 1e3, svm_long, levels=30, cmap='plasma')
    plt.colorbar(cf2, ax=axes[1]).set_label("σ_vm (Pa)")
    axes[1].set_title("Longitudinal full cross-section: von Mises stress σ_vm")
    axes[1].set_xlabel("z (mm)")
    axes[1].set_ylabel("x (mm)")

    plt.tight_layout()
    plt.savefig("inference_longitudinal_full_section.png", dpi=150)
    plt.show()
    print("Saved → inference_longitudinal_full_section.png")

    # 3D point cloud coloured by displacement
    all_pts_3d = np.vstack([nodes_ok, centres_ok])
    all_umag_3d = np.concatenate([node_fields['umag'], centre_fields['umag']]) * 1e6
    node_bnd = mask_inn | mask_out | mask_z0 | mask_zL
    all_sz_3d = np.concatenate([np.where(node_bnd, 8, 2), np.full(len(centres_ok), 2)])

    fig = plt.figure(figsize=(10, 8))
    ax_d = fig.add_subplot(111, projection='3d')
    sc = ax_d.scatter(
        all_pts_3d[:, 0] * 1e3,
        all_pts_3d[:, 1] * 1e3,
        all_pts_3d[:, 2] * 1e3,
        c=all_umag_3d, cmap='plasma', s=all_sz_3d, alpha=0.6, linewidths=0
    )
    plt.colorbar(sc, ax=ax_d, pad=0.12, shrink=0.6).set_label("|u| (µm)", fontsize=11)
    ax_d.set_xlabel("x1 (mm)")
    ax_d.set_ylabel("x2 (mm)")
    ax_d.set_zlabel("z (mm)")
    ax_d.set_title("3D mesh points — coloured by |u|\n(z=0 → 0,  z=L → 0)", fontsize=11)

    plt.tight_layout()
    plt.savefig("inference_collocation_points_3d_disp.png", dpi=150)
    plt.show()
    print("Saved → inference_collocation_points_3d_disp.png")

    # MATLAB export
    print("\nBuilding structured surface grid for MATLAB …")
    Nth_s, Nz_s = 72, 40
    th_s = np.linspace(0, 2 * math.pi, Nth_s, endpoint=False)
    z_s  = np.linspace(0, L, Nz_s)
    TH_s, Z_s = np.meshgrid(th_s, z_s, indexing='ij')

    def surface_disp(rv):
        X_s = rv * np.cos(TH_s)
        Y_s = rv * np.sin(TH_s)
        pts = np.stack([X_s.ravel(), Y_s.ravel(), Z_s.ravel()], axis=1)
        x_t = torch.tensor(pts, dtype=torch.float64, device=device, requires_grad=True)
        with torch.enable_grad():
            u_g = model(x_t)
            _, _, _, _, _, _, _, _, svmg, _, _, _ = strains_stresses(u_g, x_t)

        def g(t):
            return t.detach().cpu().numpy().reshape(Nth_s, Nz_s)

        return dict(
            X=X_s, Y=Y_s, Z=Z_s,
            u1=g(u_g[:, 0:1]), u2=g(u_g[:, 1:2]), u3=g(u_g[:, 2:3]),
            umag=np.sqrt((u_g.detach().cpu().numpy()**2).sum(1)).reshape(Nth_s, Nz_s),
            svm=g(svmg)
        )

    outer_surf = surface_disp(r_o)
    inner_surf = surface_disp(r_i)

    node_indices_ok = np.where(in_domain(nodes))[0]
    old2new = {old: new for new, old in enumerate(node_indices_ok)}
    tet_ok = []
    for tet in tet_conn:
        if all(n in old2new for n in tet):
            tet_ok.append([old2new[n] for n in tet])
    tet_ok = np.array(tet_ok, dtype=np.int64) if tet_ok else np.zeros((0, 4), dtype=np.int64)
    print(f"  Tetrahedra fully inside domain: {len(tet_ok)}")

    try:
        import scipy.io
        scipy.io.savemat("pinn_inference_results.mat", {
            "nodes": nodes_ok,
            "tet_conn": tet_ok + 1,
            "u1": node_fields['u1'],
            "u2": node_fields['u2'],
            "u3": node_fields['u3'],
            "umag": node_fields['umag'],
            "s_vm": node_fields['s_vm'],
            "s_rr": node_fields['s_rr'],
            "s_tt": node_fields['s_tt'],
            "s11": node_fields['s11'],
            "s22": node_fields['s22'],
            "s33": node_fields['s33'],
            "X_outer": outer_surf['X'], "Y_outer": outer_surf['Y'], "Z_outer": outer_surf['Z'],
            "u1_outer": outer_surf['u1'], "u2_outer": outer_surf['u2'], "u3_outer": outer_surf['u3'],
            "umag_outer": outer_surf['umag'], "svm_outer": outer_surf['svm'],
            "X_inner": inner_surf['X'], "Y_inner": inner_surf['Y'], "Z_inner": inner_surf['Z'],
            "u1_inner": inner_surf['u1'], "u2_inner": inner_surf['u2'], "u3_inner": inner_surf['u3'],
            "umag_inner": inner_surf['umag'], "svm_inner": inner_surf['svm'],
            "th_vec": th_s, "z_vec": z_s,
            "Nth_s": Nth_s, "Nz_s": Nz_s,
            "r_i": r_i, "r_o": r_o, "L": L,
            "E": E, "nu": nu,
            "p_i_mid": p_i_mid, "p_o": p_o,
            # internal pressure profile
        })
        print("Saved → pinn_inference_results.mat")
    except ImportError:
        print("scipy not found — skipping .mat  (pip install scipy)")

    # Final summary
    print(f"\n{'='*56}\n  FINAL SUMMARY  (PINN — both ends fixed)\n{'='*56}")
    for zv, zlbl in [(L/3, "z=L/3"), (0.5 * L, "z=L/2"), (3 * L / 4, "z=3L/4")]:
        d = eval_grid(zv)
        print(f"\n  At {zlbl}:")
        print(f"    p_i(z)   = {p_i_of_z(zv):.3e} Pa")
        print(f"    Max σ_vm = {d['svm'].max():.3e} Pa")
        print(f"    Max |u|  = {d['umag'].max():.3e} m")

    print(f"\n  p_i: cosine  p_i(0)=0  p_i(L/2)={p_i_mid:.1f} Pa  p_i(L)=0")
    print(f"  p_o: uniform = {p_o:.1f} Pa")
    print(f"{'='*56}")
    print("\nDone. Load pinn_inference_results.mat in MATLAB with the provided script.")


if __name__ == '__main__':
    main()