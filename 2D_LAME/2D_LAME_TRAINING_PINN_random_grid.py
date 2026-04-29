import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt

# ============================================================
# 1. Problem setup and material parameters
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Artery-like geometry used for the 2D Lamé verification case
r_i = 0.0015      # 1.5 mm
t   = 0.0002      # 0.2 mm wall thickness
r_o = r_i + t     # 1.7 mm

E   = 5e5         # Young's modulus
nu  = 0.45

p_i = 100 * 133.322   # 100 mmHg -> Pa
p_o = 10  * 133.322   # 10 mmHg  -> Pa

# Lamé parameters used in the plane strain formulation
mu  = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))

# ============================================================
# 2. Analytical Lamé solution for radial displacement
# ============================================================

def lame_constants_from_pressures(ri, ro, pi, po):
    B = (pi - po) / (1.0 / ri**2 - 1.0 / ro**2)
    A = -po + B / ro**2
    return A, B

A_lame, B_lame = lame_constants_from_pressures(r_i, r_o, p_i, p_o)

A_t = torch.tensor(A_lame, dtype=torch.float64, device=device)
B_t = torch.tensor(B_lame, dtype=torch.float64, device=device)

def u_r_exact(r):
    r = np.asarray(r)
    return ((1.0 + nu) / E) * ((1.0 - 2.0*nu) * A_lame * r + B_lame / r)

# ============================================================
# 3. Random collocation points and boundary points in the quarter annulus
# ============================================================

# -------------------------------
# Number of points used for the interior and boundary sets
# -------------------------------
N_int = 50      # interior collocation points
N_bnd = 10       # points on each circular boundary
N_ax  = 10       # points on each symmetry axis

# Fixed random seed for repeatable point generation
np.random.seed(0)
torch.manual_seed(0)

# -------------------------------
# Interior points sampled uniformly over the annular area
# -------------------------------
u = np.random.rand(N_int)
r = np.sqrt(r_i**2 + (r_o**2 - r_i**2) * u)                 # area-uniform
theta = (0.5 * np.pi) * np.random.rand(N_int)               # uniform angle in [0, pi/2]

x_int_np = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

# -------------------------------
# Inner boundary at r = r_i with pressure traction
# -------------------------------
theta_in = (0.5 * np.pi) * np.random.rand(N_bnd)
x_inn_np = np.stack([r_i * np.cos(theta_in), r_i * np.sin(theta_in)], axis=1)

# -------------------------------
# Outer boundary at r = r_o with pressure traction
# -------------------------------
theta_out = (0.5 * np.pi) * np.random.rand(N_bnd)
x_out_np = np.stack([r_o * np.cos(theta_out), r_o * np.sin(theta_out)], axis=1)

# -------------------------------
# Symmetry axes:
# x2 = 0  -> enforce u2 = 0
# x1 = 0  -> enforce u1 = 0
# Radius is sampled only within the wall thickness
# -------------------------------
r_ax = np.random.rand(N_ax)
r_ax = r_i + (r_o - r_i) * r_ax

x_x1_np = np.stack([r_ax, np.zeros_like(r_ax)], axis=1)     # x2 = 0
x_x2_np = np.stack([np.zeros_like(r_ax), r_ax], axis=1)     # x1 = 0

print(f"Interior points   : {x_int_np.shape[0]}")
print(f"Inner boundary    : {x_inn_np.shape[0]}")
print(f"Outer boundary    : {x_out_np.shape[0]}")
print(f"x2 = 0 axis points: {x_x1_np.shape[0]}")
print(f"x1 = 0 axis points: {x_x2_np.shape[0]}")

# Analytical boundary displacement values used for normalization
r_inn = np.sqrt(x_inn_np[:, 0]**2 + x_inn_np[:, 1]**2)
r_out = np.sqrt(x_out_np[:, 0]**2 + x_out_np[:, 1]**2)

u_r_inn_exact = u_r_exact(r_inn)
u_r_out_exact = u_r_exact(r_out)

# ============================================================
# 4. Conversion from NumPy arrays to PyTorch tensors
# ============================================================

def to_tensor(x, requires_grad=False):
    return torch.tensor(x, dtype=torch.float64, requires_grad=requires_grad, device=device)

x_int  = to_tensor(x_int_np,  requires_grad=True)
x_inn  = to_tensor(x_inn_np,  requires_grad=True)
x_out  = to_tensor(x_out_np,  requires_grad=True)
x_x1   = to_tensor(x_x1_np,   requires_grad=True)
x_x2   = to_tensor(x_x2_np,   requires_grad=True)

u_scale = float(max(np.max(np.abs(u_r_inn_exact)), np.max(np.abs(u_r_out_exact))))

# ============================================================
# 5. PINN architecture
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
        if x.shape[1] == 2:
            x1 = x[:, 0:1]
            x2 = x[:, 1:2]
            r = torch.sqrt(x1**2 + x2**2)
        else:
            r = x
            x1 = r
            x2 = torch.zeros_like(r)

        r_norm = 2.0 * (r - r_i) / (r_o - r_i) - 1.0

        z = self.activation(self.input_layer(r_norm))
        for layer in self.hidden_layers:
            z = self.activation(layer(z))

        u_r = self.output_layer(z)  # displacement is learned directly from the network output

        eps = 1e-12
        nx = x1 / (r + eps)
        ny = x2 / (r + eps)

        u1 = u_r * nx
        u2 = u_r * ny
        return torch.cat([u1, u2], dim=1)

model = LamePINN().to(device)

# ============================================================
# 6. Automatic differentiation utilities
# ============================================================

def grad(u_comp, x):
    return torch.autograd.grad(
        u_comp, x,
        grad_outputs=torch.ones_like(u_comp),
        create_graph=True
    )[0]

def laplacian(u_comp, x):
    g = grad(u_comp, x)
    g1 = grad(g[:, 0:1], x)
    g2 = grad(g[:, 1:2], x)
    return g1[:, 0:1] + g2[:, 1:2]

def traction_on_points(model, x, p_target):
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)

    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    r  = torch.sqrt(x1**2 + x2**2) + 1e-12

    u = model(x)
    u1 = u[:, 0:1]
    u2 = u[:, 1:2]

    grad_u1 = grad(u1, x)
    grad_u2 = grad(u2, x)

    u1_x1 = grad_u1[:, 0:1]
    u1_x2 = grad_u1[:, 1:2]
    u2_x1 = grad_u2[:, 0:1]
    u2_x2 = grad_u2[:, 1:2]

    eps11 = u1_x1
    eps22 = u2_x2
    eps12 = 0.5 * (u1_x2 + u2_x1)

    trace_eps = eps11 + eps22

    sigma11 = lam * trace_eps + 2.0 * mu * eps11
    sigma22 = lam * trace_eps + 2.0 * mu * eps22
    sigma12 = 2.0 * mu * eps12

    n1 = x1 / r
    n2 = x2 / r

    t1 = sigma11 * n1 + sigma12 * n2
    t2 = sigma12 * n1 + sigma22 * n2

    p_t = torch.tensor(p_target, dtype=torch.float64, device=device)
    t1_target = -p_t * n1
    t2_target = -p_t * n2

    res1 = t1 - t1_target
    res2 = t2 - t2_target
    return res1, res2

def stresses_on_points(model, x):
    """
    Evaluate Cartesian, radial, hoop and von Mises stresses at the given points.
    """
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)

    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    r  = torch.sqrt(x1**2 + x2**2) + 1e-12

    u = model(x)
    u1 = u[:, 0:1]
    u2 = u[:, 1:2]

    grad_u1 = grad(u1, x)
    grad_u2 = grad(u2, x)

    u1_x1 = grad_u1[:, 0:1]
    u1_x2 = grad_u1[:, 1:2]
    u2_x1 = grad_u2[:, 0:1]
    u2_x2 = grad_u2[:, 1:2]

    eps11 = u1_x1
    eps22 = u2_x2
    eps12 = 0.5 * (u1_x2 + u2_x1)

    tr = eps11 + eps22

    sigma11 = lam * tr + 2.0 * mu * eps11
    sigma22 = lam * tr + 2.0 * mu * eps22
    sigma12 = 2.0 * mu * eps12

    c = x1 / r
    s = x2 / r

    sigma_rr = sigma11*c*c + 2.0*sigma12*c*s + sigma22*s*s
    sigma_tt = sigma11*s*s - 2.0*sigma12*c*s + sigma22*c*c

    sigma_vm = torch.sqrt(sigma11**2 - sigma11*sigma22 + sigma22**2 + 3.0*sigma12**2)

    return sigma11, sigma22, sigma12, sigma_rr, sigma_tt, sigma_vm

# ============================================================
# 7. PINN loss function
# ============================================================

def loss_pinn(model):
    x = x_int

    u = model(x)
    u1 = u[:, 0:1]
    u2 = u[:, 1:2]

    grad_u1 = grad(u1, x)
    grad_u2 = grad(u2, x)

    u1_x1 = grad_u1[:, 0:1]
    u1_x2 = grad_u1[:, 1:2]
    u2_x1 = grad_u2[:, 0:1]
    u2_x2 = grad_u2[:, 1:2]

    div_u = u1_x1 + u2_x2

    grad_div = grad(div_u, x)
    div_x1 = grad_div[:, 0:1]
    div_x2 = grad_div[:, 1:2]

    lap_u1 = laplacian(u1, x)
    lap_u2 = laplacian(u2, x)

    Du1 = (lam + mu) * div_x1 + mu * lap_u1
    Du2 = (lam + mu) * div_x2 + mu * lap_u2

    loss_eq = torch.mean((Du1 / p_i)**2 + (Du2 / p_i)**2)

    if x_inn.shape[0] > 0:
        t1_inn_res, t2_inn_res = traction_on_points(model, x_inn, p_i)
        loss_inn = torch.mean((t1_inn_res / p_i)**2 + (t2_inn_res / p_i)**2)
    else:
        loss_inn = torch.tensor(0.0, dtype=torch.float64, device=device)

    if x_out.shape[0] > 0:
        t1_out_res, t2_out_res = traction_on_points(model, x_out, p_o)
        loss_out = torch.mean((t1_out_res / p_i)**2 + (t2_out_res / p_i)**2)
    else:
        loss_out = torch.tensor(0.0, dtype=torch.float64, device=device)

    if x_x1.shape[0] > 0:
        u_x1 = model(x_x1)
        u2_x1_val = u_x1[:, 1:2]
        loss_sym_x1 = torch.mean((u2_x1_val / u_scale)**2)
    else:
        loss_sym_x1 = torch.tensor(0.0, dtype=torch.float64, device=device)

    if x_x2.shape[0] > 0:
        u_x2 = model(x_x2)
        u1_x2_val = u_x2[:, 0:1]
        loss_sym_x2 = torch.mean((u1_x2_val / u_scale)**2)
    else:
        loss_sym_x2 = torch.tensor(0.0, dtype=torch.float64, device=device)

    w_eq   = 1.0
    w_bc   = 100.0
    w_sym  = 100.0

    total = (w_eq * loss_eq +
             w_bc * (loss_inn + loss_out) +
             w_sym * (loss_sym_x1 + loss_sym_x2))

    return total, {
        "eq": loss_eq.detach().cpu().item(),
        "inn": loss_inn.detach().cpu().item(),
        "out": loss_out.detach().cpu().item(),
        "sym_x1": loss_sym_x1.detach().cpu().item(),
        "sym_x2": loss_sym_x2.detach().cpu().item()
    }

# ============================================================
# 8. Model training
# ============================================================

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min',
    factor=0.5, patience=200,
    verbose=True
)

num_epochs = 3000
history = {"loss": [], "eq": [], "inn": [], "out": [], "sym_x1": [], "sym_x2": []}

print("\n" + "="*60)
print("Starting Adam optimization...")
print("="*60)

for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()
    loss, comp = loss_pinn(model)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(loss)

    if not torch.isfinite(loss):
        print("Loss became NaN/Inf at epoch", epoch)
        break

    history["loss"].append(loss.item())
    for k in comp:
        history[k].append(comp[k])

    if epoch % 500 == 0 or epoch == 1:
        print(f"Epoch {epoch:5d}  total={loss.item():.3e}  "
              f"eq={comp['eq']:.3e}  inn={comp['inn']:.3e}  out={comp['out']:.3e}")

print("\n" + "="*60)
print("Starting L-BFGS refinement...")
print("="*60)

optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=500,
    tolerance_grad=1e-12,
    tolerance_change=1e-12,
    history_size=50,
    line_search_fn='strong_wolfe'
)

def closure():
    optimizer_lbfgs.zero_grad()
    loss, _ = loss_pinn(model)
    loss.backward()
    return loss

optimizer_lbfgs.step(closure)

print("\nTraining complete!")
print("="*60)

# ============================================================
# 9. Boundary traction residual check
# ============================================================

model.eval()

with torch.enable_grad():
    if x_inn.shape[0] > 0:
        t1_inn_res, t2_inn_res = traction_on_points(model, x_inn, p_i)
        err_inn_t = torch.max(torch.sqrt(t1_inn_res**2 + t2_inn_res**2)).item()
    else:
        err_inn_t = 0.0

    if x_out.shape[0] > 0:
        t1_out_res, t2_out_res = traction_on_points(model, x_out, p_o)
        err_out_t = torch.max(torch.sqrt(t1_out_res**2 + t2_out_res**2)).item()
    else:
        err_out_t = 0.0

print(f"\nmax inner traction residual: {err_inn_t:.3e}")
print(f"max outer traction residual: {err_out_t:.3e}")

# ============================================================
# 10. Comparison between analytical and PINN displacement components
# ============================================================

r_plot = np.linspace(r_i, r_o, 200)
u_r_ana = u_r_exact(r_plot)

theta_list = [0.0, np.pi/6, np.pi/4, np.pi/3]
theta_deg  = [0, 30, 45, 60]

plt.figure(figsize=(14, 5))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

for th, th_deg in zip(theta_list, theta_deg):
    x1_vals = r_plot * np.cos(th)
    x2_vals = r_plot * np.sin(th)
    x_plot  = np.stack([x1_vals, x2_vals], axis=1)
    x_plot_t = to_tensor(x_plot, requires_grad=False)

    with torch.no_grad():
        u_pinn = model(x_plot_t).cpu().numpy()

    u1_pinn = u_pinn[:, 0]
    u2_pinn = u_pinn[:, 1]

    u1_ana = u_r_ana * np.cos(th)
    u2_ana = u_r_ana * np.sin(th)

    ax1.plot(r_plot, u1_ana, "o", markersize=3, alpha=0.5, label=f"Analytic θ={th_deg}°")
    ax1.plot(r_plot, u1_pinn, "-", linewidth=2, label=f"PINN θ={th_deg}°")

    ax2.plot(r_plot, u2_ana, "o", markersize=3, alpha=0.5, label=f"Analytic θ={th_deg}°")
    ax2.plot(r_plot, u2_pinn, "-", linewidth=2, label=f"PINN θ={th_deg}°")

ax1.set_xlabel("r (m)")
ax1.set_ylabel("u1 (m)")
ax1.set_title("u1 component")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8, loc='best')

ax2.set_xlabel("r (m)")
ax2.set_ylabel("u2 (m)")
ax2.set_title("u2 component")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8, loc='best')

plt.tight_layout()
plt.show()

# ============================================================
# 10b. Training loss history
# ============================================================

plt.figure(figsize=(10, 8))

ax_eq = plt.subplot(3, 1, 1)
ax_eq.semilogy(history["eq"])
ax_eq.set_ylabel("PDE loss")
ax_eq.set_title("PDE residual loss")
ax_eq.grid(True, alpha=0.3)

ax_bc = plt.subplot(3, 1, 2)
ax_bc.semilogy(history["inn"], label="inner")
ax_bc.semilogy(history["out"], label="outer")
ax_bc.set_ylabel("BC loss")
ax_bc.set_title("Traction BC losses")
ax_bc.legend()
ax_bc.grid(True, alpha=0.3)

ax_sym = plt.subplot(3, 1, 3)
ax_sym.semilogy(history["sym_x1"], label="x2 = 0")
ax_sym.semilogy(history["sym_x2"], label="x1 = 0")
ax_sym.set_xlabel("Epoch")
ax_sym.set_ylabel("Symmetry loss")
ax_sym.set_title("Symmetry losses")
ax_sym.legend()
ax_sym.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# 11. Plot of collocation and boundary points
# ============================================================

plt.figure(figsize=(6, 6))

plt.scatter(x_int_np[:, 0], x_int_np[:, 1], s=5, alpha=0.6, label="Interior")
plt.scatter(x_inn_np[:, 0], x_inn_np[:, 1], s=18, label="Inner boundary")
plt.scatter(x_out_np[:, 0], x_out_np[:, 1], s=18, label="Outer boundary")
plt.scatter(x_x1_np[:, 0],  x_x1_np[:, 1],  s=18, label="x2 = 0 axis")
plt.scatter(x_x2_np[:, 0],  x_x2_np[:, 1],  s=18, label="x1 = 0 axis")

# Draw inner and outer arcs for reference
theta_plot = np.linspace(0.0, 0.5 * np.pi, 250)
plt.plot(r_i*np.cos(theta_plot), r_i*np.sin(theta_plot), "k", linewidth=2)
plt.plot(r_o*np.cos(theta_plot), r_o*np.sin(theta_plot), "k", linewidth=2)

plt.gca().set_aspect("equal", "box")
plt.xlabel("x1 (m)")
plt.ylabel("x2 (m)")
plt.title("Random collocation points in quarter annulus")
plt.grid(True, alpha=0.2)
plt.legend(markerscale=1.2, fontsize=9)
plt.tight_layout()
plt.show()

# ============================================================
# 12. Displacement magnitude contour plot
# ============================================================

Nq = 100
x1_lin = np.linspace(0.0, r_o, Nq)
x2_lin = np.linspace(0.0, r_o, Nq)
Xq, Yq = np.meshgrid(x1_lin, x2_lin)

X_flat = Xq.flatten()
Y_flat = Yq.flatten()
R_flat = np.sqrt(X_flat**2 + Y_flat**2)
maskq = (R_flat >= r_i) & (R_flat <= r_o)

xq_np = np.stack([X_flat[maskq], Y_flat[maskq]], axis=1)

with torch.no_grad():
    xq_t = to_tensor(xq_np, requires_grad=False)
    uq = model(xq_t).cpu().numpy()

U = np.zeros_like(X_flat)
V = np.zeros_like(Y_flat)
U[maskq] = uq[:, 0]
V[maskq] = uq[:, 1]

U = U.reshape(Xq.shape)
V = V.reshape(Yq.shape)

disp_mag = np.sqrt(U**2 + V**2)

plt.figure(figsize=(6, 6))
cont = plt.contourf(Xq, Yq, disp_mag, levels=30, cmap="viridis")
plt.colorbar(cont, label="|u| (m)")

theta_plot = np.linspace(0.0, 0.5 * math.pi, 200)
plt.plot(r_i*np.cos(theta_plot), r_i*np.sin(theta_plot), "k", linewidth=2)
plt.plot(r_o*np.cos(theta_plot), r_o*np.sin(theta_plot), "k", linewidth=2)

plt.gca().set_aspect("equal", "box")
plt.xlabel("x1 (m)")
plt.ylabel("x2 (m)")
plt.title("Displacement magnitude |u| in quarter annulus")
plt.grid(True, alpha=0.2)
plt.show()

'''
# ============================================================
# 13. Stress line plot along the radial direction
# ============================================================

r_line = np.linspace(r_i, r_o, 400)
theta_test = np.pi/4

x_line = np.stack([r_line*np.cos(theta_test), r_line*np.sin(theta_test)], axis=1)
x_line_t = torch.tensor(x_line, dtype=torch.float64, device=device, requires_grad=True)

sigma11, sigma22, sigma12, sigma_rr, sigma_tt, sigma_vm = stresses_on_points(model, x_line_t)

sigma_rr_np = sigma_rr.detach().cpu().numpy().flatten()
sigma_tt_np = sigma_tt.detach().cpu().numpy().flatten()
sigma_vm_np = sigma_vm.detach().cpu().numpy().flatten()

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(r_line, sigma_rr_np, linewidth=2, label=r"$\sigma_{rr}$")
plt.plot(r_line, sigma_tt_np, linewidth=2, label=r"$\sigma_{\theta\theta}$")
plt.axhline(-p_i, linestyle="--", linewidth=1.5, label=r"$-p_i$")
plt.axhline(-p_o, linestyle="--", linewidth=1.5, label=r"$-p_o$")
plt.xlabel("r (m)")
plt.ylabel("Stress (Pa)")
plt.title(r"Stresses vs r at $\theta=\pi/4$")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)

plt.subplot(1, 2, 2)
plt.plot(r_line, sigma_vm_np, linewidth=2, label=r"$\sigma_{vm}$")
plt.xlabel("r (m)")
plt.ylabel("Stress (Pa)")
plt.title("von Mises vs r")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
'''

# ============================================================
# 14. Check of radial stress at the inner and outer boundaries
# ============================================================

with torch.enable_grad():
    # Inner boundary stresses
    s11_i, s22_i, s12_i, srr_i, stt_i, svm_i = stresses_on_points(model, x_inn)
    # Outer boundary stresses
    s11_o, s22_o, s12_o, srr_o, stt_o, svm_o = stresses_on_points(model, x_out)

srr_i_np = srr_i.detach().cpu().numpy().flatten()
srr_o_np = srr_o.detach().cpu().numpy().flatten()

print("\nSANITY CHECK sigma_rr boundary values")
print(f"Inner: mean sigma_rr = {np.mean(srr_i_np):.3e} Pa, target = {-p_i:.3e} Pa")
print(f"Inner:  max abs err  = {np.max(np.abs(srr_i_np + p_i)):.3e} Pa")
print(f"Outer: mean sigma_rr = {np.mean(srr_o_np):.3e} Pa, target = {-p_o:.3e} Pa")
print(f"Outer:  max abs err  = {np.max(np.abs(srr_o_np + p_o)):.3e} Pa")

# ============================================================
# 15. von Mises stress contour plot
# ============================================================

# Use the same grid as the displacement contour plot
xq_t2 = torch.tensor(xq_np, dtype=torch.float64, device=device, requires_grad=True)

# Compute stresses and extract the von Mises stress field
_, _, _, _, _, svm = stresses_on_points(model, xq_t2)

svm = svm.detach().cpu().numpy().flatten()

# Map values back to the full plotting grid
SVM = np.zeros_like(X_flat)
SVM[maskq] = svm
SVM = SVM.reshape(Xq.shape)

# Boundary curves used in the contour plot
theta_plot = np.linspace(0.0, 0.5 * math.pi, 200)
inner_x = r_i * np.cos(theta_plot); inner_y = r_i * np.sin(theta_plot)
outer_x = r_o * np.cos(theta_plot); outer_y = r_o * np.sin(theta_plot)

# Plot the von Mises stress field
plt.figure(figsize=(6, 6))
cont = plt.contourf(Xq, Yq, SVM, levels=40, cmap="viridis")
plt.colorbar(cont, label=r"$\sigma_{vm}$ (Pa)")
plt.plot(inner_x, inner_y, "k", linewidth=2)
plt.plot(outer_x, outer_y, "k", linewidth=2)
plt.gca().set_aspect("equal", "box")
plt.xlabel("x1 (m)")
plt.ylabel("x2 (m)")
plt.title("von Mises stress in quarter annulus")
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

# ============================================================
# 16. Maximum stress and displacement values
# ============================================================

model.eval()

# Maximum von Mises stress in the domain
x_dom = torch.tensor(xq_np, dtype=torch.float64, device=device, requires_grad=True)
_, _, _, _, _, svm_dom = stresses_on_points(model, x_dom)

max_svm = torch.max(svm_dom).item()

# Maximum radial stress magnitude at the inner wall
with torch.enable_grad():
    _, _, _, srr_inn, _, _ = stresses_on_points(model, x_inn)

max_abs_srr_inn = torch.max(torch.abs(srr_inn)).item()

# Also report the most compressive radial stress value
min_srr_inn = torch.min(srr_inn).item()

# Maximum displacement magnitude in the domain
# Gradients are not required for displacement evaluation
with torch.no_grad():
    x_dom_ng = torch.tensor(xq_np, dtype=torch.float64, device=device, requires_grad=False)
    u_dom = model(x_dom_ng)              # (N,2)
    disp_mag_dom = torch.sqrt(u_dom[:,0:1]**2 + u_dom[:,1:2]**2)

max_disp = torch.max(disp_mag_dom).item()

print("\n" + "="*60)
print("FINAL MAX VALUES (PINN)")
print("="*60)
print(f"max von Mises stress in domain        = {max_svm:.3e} Pa")
print(f"max |sigma_rr| at inner wall (r=r_i)  = {max_abs_srr_inn:.3e} Pa")
print(f"min sigma_rr at inner wall = {min_srr_inn:.3e} Pa")
print(f"max displacement magnitude in domain  = {max_disp:.3e} m")
print("="*60)
