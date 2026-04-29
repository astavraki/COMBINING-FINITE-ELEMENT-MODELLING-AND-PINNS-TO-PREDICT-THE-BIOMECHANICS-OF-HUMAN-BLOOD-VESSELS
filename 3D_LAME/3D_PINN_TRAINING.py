"""
Training script for the 3D hollow cylinder under uniform pressure.

The code solves the Navier-Cauchy equations for linear elasticity using a
Physics-Informed Neural Network. The inner and outer cylindrical surfaces are
loaded with pressure tractions, while the axial displacement is constrained at
the two ends through the network architecture.

The trained model is saved and then used for inference and comparison with the
analytical Lamé solution.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import json, os

# ============================================================
# User-defined parameters
# ============================================================

# Geometry and material parameters
r_i = 0.0015      # inner radius (m)
t   = 0.0002      # wall thickness (m)
L   = 0.02        # pipe length (m)
E   = 5e5         # Young's modulus (Pa)
nu  = 0.45        # Poisson's ratio
p_i = 100 * 133.322   # inner pressure (Pa)
p_o = 10  * 133.322   # outer pressure (Pa)

# Collocation points
N_interior     = 2688  # interior points
N_wall         = 384   # points per wall surface
N_end          = 224   # points per end cap
resample_every = 500   # resampling interval

# Training settings
num_epochs  = 5000   # number of training epochs
bc_switch   = 1500   # transition from Phase A to Phase B

# ============================================================
# Main implementation
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
r_o = r_i + t
mu  = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))
pde_scale = p_i / t

print(f"Device      : {device}")
print(f"mu={mu:.3e} Pa  lam={lam:.3e} Pa  pde_scale={pde_scale:.3e} Pa/m")
print(f"p_i={p_i:.3e} Pa  p_o={p_o:.3e} Pa")

# ============================================================
# 1. Analytical Lamé solution
# ============================================================

def lame_constants(ri, ro, pi, po):
    B = (pi - po) / (1.0/ri**2 - 1.0/ro**2)
    A = -po + B / ro**2
    return A, B

A_lame, B_lame = lame_constants(r_i, r_o, p_i, p_o)

def u_r_exact(r):
    r = np.asarray(r)
    return ((1+nu)/E) * ((1-2*nu)*A_lame*r + B_lame/r)

def sigma_rr_exact(r): return A_lame - B_lame/np.asarray(r)**2
def sigma_tt_exact(r): return A_lame + B_lame/np.asarray(r)**2
def sigma_zz_exact(r): return np.full_like(np.asarray(r, float), 2*nu*A_lame)
def sigma_vm_exact(r):
    s, t_, z = sigma_rr_exact(r), sigma_tt_exact(r), sigma_zz_exact(r)
    return np.sqrt(0.5*((s-t_)**2+(t_-z)**2+(z-s)**2))

u_scale         = float(max(abs(u_r_exact(r_i)), abs(u_r_exact(r_o))))
szz_exact_const = float(2.0*nu*A_lame)
print(f"u_scale={u_scale:.3e} m   σ_zz_exact={szz_exact_const:.3e} Pa")

# ============================================================
# 2. Collocation points
# ============================================================

N_interior    = 2688   # interior collocation points  (≈ same as structured)
N_wall        = 384    # points per wall surface (inner/outer)
N_end         = 224    # points per end cap (z=0 / z=L)
resample_every = 500   # resampling interval

rng = np.random.default_rng(42)   # fixed seed

def to_tensor(arr, grad=False):
    return torch.tensor(arr, dtype=torch.float64,
                        requires_grad=grad, device=device)

def sample_interior(n, seed=None):
    """Sample points inside the annular volume."""
    g = np.random.default_rng(seed)
    r   = np.sqrt(g.uniform(r_i**2, r_o**2, n))   # uniform in area
    th  = g.uniform(0.0, 2*math.pi, n)
    z   = g.uniform(0.0, L, n)
    return np.stack([r*np.cos(th), r*np.sin(th), z], axis=1)

def sample_wall(n, rv, seed=None):
    """Sample points on a cylindrical wall."""
    g = np.random.default_rng(seed)
    th = g.uniform(0.0, 2*math.pi, n)
    z  = g.uniform(0.0, L, n)
    return np.stack([rv*np.cos(th), rv*np.sin(th), z], axis=1)

def sample_end(n, zv, seed=None):
    """Sample points on an end cap."""
    g = np.random.default_rng(seed)
    r  = np.sqrt(g.uniform(r_i**2, r_o**2, n))
    th = g.uniform(0.0, 2*math.pi, n)
    return np.stack([r*np.cos(th), r*np.sin(th), np.full(n, zv)], axis=1)

def make_tensors(seed=None):
    """Create all point sets used during training."""
    s = seed
    x_int_np = sample_interior(N_interior, s)
    x_inn_np = sample_wall(N_wall, r_i, s)
    x_out_np = sample_wall(N_wall, r_o, s)
    x_z0_np  = sample_end(N_end, 0.0, s)
    x_zL_np  = sample_end(N_end, L,   s)
    return (to_tensor(x_int_np, grad=True),
            to_tensor(x_inn_np, grad=True),
            to_tensor(x_out_np, grad=True),
            to_tensor(x_z0_np,  grad=True),
            to_tensor(x_zL_np,  grad=True))

x_int, x_inn, x_out, x_z0, x_zL = make_tensors(seed=0)

# Fixed wall points used in the traction loss
x_inn_dense = to_tensor(sample_wall(N_wall, r_i, seed=99), grad=True)
x_out_dense = to_tensor(sample_wall(N_wall, r_o, seed=100), grad=True)

# Static point set for the collocation plot
x_all_plot        = np.vstack([sample_interior(N_interior,0),
                                sample_wall(N_wall,r_i,0),
                                sample_wall(N_wall,r_o,0),
                                sample_end(N_end,0.0,0),
                                sample_end(N_end,L,0)])
interior_mask_plot = np.arange(N_interior)
inner_mask_plot    = np.arange(N_interior, N_interior+N_wall)
outer_mask_plot    = np.arange(N_interior+N_wall, N_interior+2*N_wall)
z0_mask_plot       = np.arange(N_interior+2*N_wall, N_interior+2*N_wall+N_end)
zL_mask_plot       = np.arange(N_interior+2*N_wall+N_end, len(x_all_plot))

print(f"\nRandom sampling:")
print(f"  Interior pts : {N_interior}")
print(f"  Inner wall   : {N_wall}")
print(f"  Outer wall   : {N_wall}")
print(f"  End caps     : {N_end} each")
print(f"  Resample every {resample_every} epochs")

# ============================================================
# 3. Network architecture
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
        x1 = x[:,0:1]; x2 = x[:,1:2]; x3 = x[:,2:3]
        r     = torch.sqrt(x1**2 + x2**2) + 1e-12
        cos_t = x1/r;  sin_t = x2/r
        feat  = torch.cat([2*(r-r_i)/(r_o-r_i)-1,
                           cos_t, sin_t,
                           2*x3/L-1], dim=1)
        out   = self.net(feat)
        u_r   = out[:,0:1] * u_scale
        u_th  = out[:,1:2] * u_scale
        u_z_r = out[:,2:3] * u_scale
        phi   = x3*(L-x3)/(0.5*L)**2          # axial end constraint
        u1    = u_r*cos_t - u_th*sin_t
        u2    = u_r*sin_t + u_th*cos_t
        u3    = u_z_r*phi
        return torch.cat([u1, u2, u3], dim=1)

model = PipePINN(n_hidden=6, n_neurons=80).to(device)
n_p = sum(p.numel() for p in model.parameters())
print(f"\nParameters: {n_p}")

# ============================================================
# 4. Autograd helpers
# ============================================================

def grad_vec(f, x):
    return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f),
                               create_graph=True)[0]

def diag_hessian(g_col, x):
    return torch.autograd.grad(g_col, x, grad_outputs=torch.ones_like(g_col),
                               create_graph=True)[0]

def laplacian_from_grads(g1, g2, g3, x):
    """Compute the Laplacian terms from first-order gradients."""
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
    u1=u[:,0:1]; u2=u[:,1:2]; u3=u[:,2:3]
    g1=grad_vec(u1,x); g2=grad_vec(u2,x); g3=grad_vec(u3,x)
    e11=g1[:,0:1]; e22=g2[:,1:2]; e33=g3[:,2:3]
    e12=0.5*(g1[:,1:2]+g2[:,0:1])
    e13=0.5*(g1[:,2:3]+g3[:,0:1])
    e23=0.5*(g2[:,2:3]+g3[:,1:2])
    tr=e11+e22+e33
    s11=lam*tr+2*mu*e11; s22=lam*tr+2*mu*e22; s33=lam*tr+2*mu*e33
    s12=2*mu*e12; s13=2*mu*e13; s23=2*mu*e23
    r=torch.sqrt(x[:,0:1]**2+x[:,1:2]**2)+1e-12
    c=x[:,0:1]/r; s_=x[:,1:2]/r
    s_rr = s11*c**2  + 2*s12*c*s_  + s22*s_**2
    s_tt = s11*s_**2 - 2*s12*c*s_  + s22*c**2
    s_vm = torch.sqrt(0.5*((s11-s22)**2+(s22-s33)**2+(s33-s11)**2
                           +6*(s12**2+s13**2+s23**2)))
    return s11,s22,s33,s12,s13,s23,s_rr,s_tt,s_vm

def traction_residual(mdl, x, p_target):
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)
    x1=x[:,0:1]; x2=x[:,1:2]
    r=torch.sqrt(x1**2+x2**2)+1e-12
    n1=x1/r; n2=x2/r
    u=mdl(x)
    s11,s22,s33,s12,s13,s23,_,_,_=strains_stresses(u,x)
    pt=torch.tensor(p_target, dtype=torch.float64, device=device)
    r1=(s11*n1+s12*n2)-(-pt*n1)
    r2=(s12*n1+s22*n2)-(-pt*n2)
    r3= s13*n1+s23*n2
    return r1, r2, r3

# ============================================================
# 5. Loss function
# ============================================================

w_bc_global = 1000.0   # modified by curriculum

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
    r1,r2,r3=traction_residual(mdl, x_inn_dense, p_i)
    loss_inn=torch.mean((r1/p_i)**2+(r2/p_i)**2+(r3/p_i)**2)
    r1,r2,r3=traction_residual(mdl, x_out_dense, p_o)
    loss_out=torch.mean((r1/p_i)**2+(r2/p_i)**2+(r3/p_i)**2)

    # Axial displacement at the end faces
    loss_z0=torch.mean((mdl(x_z0)[:,2:3]/u_scale)**2)
    loss_zL=torch.mean((mdl(x_zL)[:,2:3]/u_scale)**2)

    # Additional stress penalties for the uniform-pressure case
    e11=g1[:,0:1]; e22=g2[:,1:2]; e33=g3[:,2:3]
    e13=0.5*(g1[:,2:3]+g3[:,0:1]); e23=0.5*(g2[:,2:3]+g3[:,1:2])
    tr=e11+e22+e33
    s11i=lam*tr+2*mu*e11; s22i=lam*tr+2*mu*e22; s33i=lam*tr+2*mu*e33
    s13i=2*mu*e13; s23i=2*mu*e23
    loss_shear=torch.mean((s13i/p_i)**2+(s23i/p_i)**2)
    loss_szz  =torch.mean(((s33i-nu*(s11i+s22i))/p_i)**2)

    total = (1.0   * loss_eq
           + w_bc_global * (loss_inn + loss_out)
           + 10.0  * (loss_z0 + loss_zL)
           + 50.0  * loss_shear
           + 200.0 * loss_szz)

    comp = dict(eq=loss_eq.item(), inn=loss_inn.item(), out=loss_out.item(),
                z0=loss_z0.item(), zL=loss_zL.item(),
                shear=loss_shear.item(), szz=loss_szz.item())
    return total, comp

# ============================================================
# 6. Training procedure
# ============================================================

history = {k: [] for k in ["loss","eq","inn","out","z0","zL","shear","szz"]}

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=300, verbose=True)

print(f"\n{'='*60}")
print(f"Phase A: Adam  w_bc=1000  epochs 1–{bc_switch}")
print(f"{'='*60}")

for epoch in range(1, num_epochs+1):

    if epoch == bc_switch+1:
        w_bc_global = 50.0
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=300, verbose=True)
        print(f"\n{'='*60}")
        print(f"Phase B: Adam  w_bc=50  epochs {bc_switch+1}–{num_epochs}")
        print(f"{'='*60}")

    # Resample collocation points periodically
    if epoch % resample_every == 0:
        x_int, x_inn, x_out, x_z0, x_zL = make_tensors(seed=epoch)

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
              f"szz={comp['szz']:.3e}  w_bc={w_bc_global:.0f}")

# L-BFGS refinement
print(f"\n{'='*60}")
print("Phase C: L-BFGS polish")
print(f"{'='*60}")
w_bc_global = 50.0
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
# 7. Save model weights
# ============================================================

tag = f"Ni{N_interior}_Nw{N_wall}_ep{num_epochs}"
torch.save(model.state_dict(), f"model_{tag}.pt")
print(f"\nModel saved: model_{tag}.pt")

# ============================================================
# 8. Convergence metrics
# ============================================================

model.eval()
Nr_cv, Nth_cv = 60, 120
r_cv  = np.linspace(r_i, r_o, Nr_cv)
th_cv = np.linspace(0.0, 2*math.pi, Nth_cv, endpoint=False)
R_cv, TH_cv = np.meshgrid(r_cv, th_cv, indexing='ij')
xq_cv = np.stack([(R_cv*np.cos(TH_cv)).ravel(),
                  (R_cv*np.sin(TH_cv)).ravel(),
                  np.full(Nr_cv*Nth_cv, 0.5*L)], axis=1)
xq_cv_t = torch.tensor(xq_cv, dtype=torch.float64,
                        device=device, requires_grad=True)
with torch.enable_grad():
    u_cv = model(xq_cv_t)
    _,_,s33_cv,_,s13_cv,s23_cv,_,_,svm_cv = strains_stresses(u_cv, xq_cv_t)

u_mag_p = np.sqrt((u_cv.detach().cpu().numpy()**2).sum(1)).reshape(Nr_cv,Nth_cv)
svm_p   = svm_cv.detach().cpu().numpy().reshape(Nr_cv,Nth_cv)
szz_p   = s33_cv.detach().cpu().numpy().reshape(Nr_cv,Nth_cv)
s13_p   = s13_cv.detach().cpu().numpy().reshape(Nr_cv,Nth_cv)
s23_p   = s23_cv.detach().cpu().numpy().reshape(Nr_cv,Nth_cv)

def rel_L2(pred,exact):
    return np.sqrt(np.mean((pred-exact)**2))/np.sqrt(np.mean(exact**2))

err_u   = rel_L2(u_mag_p,  u_r_exact(R_cv))
err_svm = rel_L2(svm_p,    sigma_vm_exact(R_cv))
err_szz = abs(np.mean(szz_p)-szz_exact_const)/abs(szz_exact_const)
err_shr = np.sqrt(np.mean(s13_p**2+s23_p**2))/p_i

print(f"\n{'='*60}")
print(f"METRICS  Ni={N_interior}  Nw={N_wall}  epochs={num_epochs}")
print(f"{'='*60}")
print(f"  Rel L2 |u|   : {err_u  :.4e}")
print(f"  Rel L2 σ_vm  : {err_svm:.4e}")
print(f"  Rel err σ_zz : {err_szz:.4e}")
print(f"  Norm shear   : {err_shr:.4e}")
print(f"{'='*60}")

metrics = dict(N_interior=N_interior, N_wall=N_wall, N_end=N_end,
               resample_every=resample_every,
               num_epochs=num_epochs,
               n_wall_dense=int(x_inn_dense.shape[0]),
               err_u=float(err_u), err_svm=float(err_svm),
               err_szz=float(err_szz), err_shr=float(err_shr))
with open(f"metrics_{tag}.json","w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved: metrics_{tag}.json")

# ============================================================
# 9. Boundary-condition checks
# ============================================================

with torch.enable_grad():
    r1i,r2i,r3i = traction_residual(model, x_inn, p_i)
    r1o,r2o,r3o = traction_residual(model, x_out, p_o)
    err_inn = torch.max(torch.sqrt(r1i**2+r2i**2+r3i**2)).item()
    err_out = torch.max(torch.sqrt(r1o**2+r2o**2+r3o**2)).item()
with torch.no_grad():
    uz0 = model(x_z0)[:,2].cpu().numpy()
    uzL = model(x_zL)[:,2].cpu().numpy()
print(f"\nMax inner traction residual: {err_inn:.3e} Pa  (p_i={p_i:.3e})")
print(f"Max outer traction residual: {err_out:.3e} Pa  (p_o={p_o:.3e})")
print(f"Max |u3| at z=0 : {np.max(np.abs(uz0)):.3e} m")
print(f"Max |u3| at z=L : {np.max(np.abs(uzL)):.3e} m")

# ============================================================
# 10. Plots
# ============================================================

# Loss history
fig, axes = plt.subplots(4,1,figsize=(10,12))
axes[0].semilogy(history["eq"]);   axes[0].set_title("Navier-Cauchy PDE residual")
axes[1].semilogy(history["inn"],label="inner"); axes[1].semilogy(history["out"],label="outer")
axes[1].set_title("Traction BC loss"); axes[1].legend()
axes[2].semilogy(history["shear"]); axes[2].set_title("rz-shear penalty")
axes[3].semilogy(history["szz"]);   axes[3].set_title("σ_zz plane-strain penalty")
axes[3].set_xlabel("Epoch")
for ax in axes: ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig("loss_history.png",dpi=150); plt.show()

# Midplane contour grid
Nq, Nr_q = 60, 40
r_q = np.linspace(r_i, r_o, Nr_q)
t_q = np.linspace(0.0, 2*math.pi, Nq, endpoint=False)
R_q, T_q = np.meshgrid(r_q, t_q, indexing='ij')
xq_np = np.stack([(R_q*np.cos(T_q)).ravel(),
                  (R_q*np.sin(T_q)).ravel(),
                  np.full(Nr_q*Nq, 0.5*L)], axis=1)

t_cl   = np.append(t_q, t_q[0]+2*math.pi)
R_qc, T_qc = np.meshgrid(r_q, t_cl, indexing='ij')
Xm = R_qc*np.cos(T_qc); Ym = R_qc*np.sin(T_qc)
cc = lambda a: np.concatenate([a, a[:,:1]], axis=1)

theta_plot = np.linspace(0,2*math.pi,400)
def add_walls(ax):
    for rv in [r_i,r_o]:
        ax.plot(rv*np.cos(theta_plot), rv*np.sin(theta_plot),'k',lw=1.2)

with torch.no_grad():
    u_mid = model(torch.tensor(xq_np,dtype=torch.float64,device=device)).cpu().numpy()

disp_p = cc(np.sqrt((u_mid**2).sum(1)).reshape(Nr_q,Nq))
disp_e = cc(u_r_exact(R_q))
disp_err = np.abs(disp_p - disp_e)

xq_gt = torch.tensor(xq_np,dtype=torch.float64,device=device,requires_grad=True)
u_gt  = model(xq_gt)
s11q,s22q,s33q,s12q,s13q,s23q,srrq,sttq,svmq = strains_stresses(u_gt,xq_gt)
svm_p2 = cc(svmq.detach().cpu().numpy().reshape(Nr_q,Nq))
svm_e2 = cc(sigma_vm_exact(R_q))
szz_p2 = cc(s33q.detach().cpu().numpy().reshape(Nr_q,Nq))
s13_p2 = cc(s13q.detach().cpu().numpy().reshape(Nr_q,Nq))
s23_p2 = cc(s23q.detach().cpu().numpy().reshape(Nr_q,Nq))

# Displacement and von Mises contours
fig, axes = plt.subplots(2,3,figsize=(16,10))
fig.suptitle("Displacement & von Mises at z=L/2")
datasets = [disp_e, disp_p, disp_err, svm_e2, svm_p2, np.abs(svm_p2-svm_e2)]
titles   = ["Analytic |u|","PINN |u|","Error |u|",
            "Analytic σ_vm","PINN σ_vm","Error σ_vm"]
for ax,dat,ttl in zip(axes.flat, datasets, titles):
    cf=ax.contourf(Xm,Ym,dat,levels=30,cmap='viridis')
    plt.colorbar(cf,ax=ax); add_walls(ax)
    ax.set_aspect('equal'); ax.set_title(ttl)
    ax.set_xlabel("x1 (m)"); ax.set_ylabel("x2 (m)")
plt.tight_layout(); plt.savefig("stress_contours_midplane.png",dpi=150); plt.show()


# Radial profiles at theta = pi/4
r_line = np.linspace(r_i, r_o, 400)
th_l   = math.pi/4


# Displacement compared with the analytical solution
r_plot = np.linspace(r_i, r_o, 300)
z_mid  = 0.5 * L
theta_list = [0, np.pi/4, np.pi/2, 3*np.pi/4,
              np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
theta_deg  = [0, 45, 90, 135, 180, 225, 270, 315]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Radial displacement at z=L/2={z_mid*1e3:.1f} mm (full pipe)")
for th, td in zip(theta_list, theta_deg):
    xp = np.stack([r_plot*np.cos(th), r_plot*np.sin(th),
                   np.full_like(r_plot, z_mid)], axis=1)
    with torch.no_grad():
        up = model(to_tensor(xp)).cpu().numpy()
    ax1.plot(r_plot*1e3, u_r_exact(r_plot)*np.cos(th)*1e6,
             "o", ms=2, alpha=0.4, label=f"Analytic θ={td}°")
    ax1.plot(r_plot*1e3, up[:,0]*1e6, "-", lw=1.5, label=f"PINN θ={td}°")
    ax2.plot(r_plot*1e3, u_r_exact(r_plot)*np.sin(th)*1e6,
             "o", ms=2, alpha=0.4, label=f"Analytic θ={td}°")
    ax2.plot(r_plot*1e3, up[:,1]*1e6, "-", lw=1.5, label=f"PINN θ={td}°")
for ax, lbl in zip([ax1, ax2], ["u1 (µm)", "u2 (µm)"]):
    ax.set_xlabel("r (mm)"); ax.set_ylabel(lbl); ax.set_title(f"{lbl} at z=L/2")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=6, ncol=2)
plt.tight_layout(); plt.savefig("displacement_vs_analytic.png", dpi=150); plt.show()

# Total deformation contour
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Total deformation |u| at midplane z=L/2  (full pipe)")
for ax, data, title, cbl in zip(
    axes,
    [disp_e, disp_p, disp_err],
    ["Analytic |u|", "PINN |u|", "Point-wise error"],
    ["|u| (m)", "|u| (m)", "|u_pred − u_anal| (m)"]):
    vmin = disp_e.min() if "error" not in title.lower() else None
    vmax = disp_e.max() if "error" not in title.lower() else None
    cf = ax.contourf(Xm, Ym, data, levels=30, cmap='viridis',
                     **({} if vmin is None else {"vmin": vmin, "vmax": vmax}))
    plt.colorbar(cf, ax=ax).set_label(cbl)
    add_walls(ax); ax.set_aspect('equal'); ax.set_title(title)
    ax.set_xlabel("x1 (m)"); ax.set_ylabel("x2 (m)")
plt.tight_layout(); plt.savefig("deformation_contours_midplane.png", dpi=150); plt.show()

# 3D collocation point cloud
fig = plt.figure(figsize=(9, 8))
ax3 = fig.add_subplot(111, projection='3d')
def sc3(idx, c, lbl, s):
    ax3.scatter(x_all_plot[idx,0]*1e3, x_all_plot[idx,1]*1e3,
                x_all_plot[idx,2]*1e3, s=s, c=c, label=lbl, alpha=0.4)
sc3(interior_mask_plot, 'steelblue', 'Interior',   2)
sc3(inner_mask_plot,    'red',       'Inner wall',  8)
sc3(outer_mask_plot,    'orange',    'Outer wall',  8)
sc3(z0_mask_plot,       'green',     'z=0 (fixed)', 8)
sc3(zL_mask_plot,       'purple',    'z=L (fixed)', 8)
ax3.set_xlabel("x1 (mm)"); ax3.set_ylabel("x2 (mm)"); ax3.set_zlabel("z (mm)")
ax3.set_title("3D full-pipe collocation points")
ax3.legend(markerscale=3, fontsize=8)
plt.tight_layout(); plt.savefig("collocation_points_3d.png", dpi=150); plt.show()

print(f"\nDone. Weights → model_{tag}.pt")

# ============================================================
# 11. Final summary
# ============================================================

# Evaluate peak values on a dense midplane grid
Nr_fs, Nth_fs = 80, 160
r_fs   = np.linspace(r_i, r_o, Nr_fs)
th_fs  = np.linspace(0.0, 2*math.pi, Nth_fs, endpoint=False)
R_fs, TH_fs = np.meshgrid(r_fs, th_fs, indexing='ij')
xq_fs = np.stack([(R_fs*np.cos(TH_fs)).ravel(),
                  (R_fs*np.sin(TH_fs)).ravel(),
                  np.full(Nr_fs*Nth_fs, 0.5*L)], axis=1)
xq_fs_t = torch.tensor(xq_fs, dtype=torch.float64,
                        device=device, requires_grad=True)
with torch.enable_grad():
    u_fs = model(xq_fs_t)
    s11_fs,s22_fs,s33_fs,s12_fs,s13_fs,s23_fs,srr_fs,_,svm_fs = \
        strains_stresses(u_fs, xq_fs_t)

u_mag_fs  = torch.sqrt((u_fs**2).sum(dim=1)).detach().cpu().numpy()
svm_fs_np = svm_fs.detach().cpu().numpy().ravel()
srr_fs_np = srr_fs.detach().cpu().numpy().ravel()

max_vm   = svm_fs_np.max()
max_srr  = np.abs(srr_fs_np).max()
max_umag = u_mag_fs.max()

print(f"\n{'='*52}")
print(f"  FINAL SUMMARY (PINN, 3D FULL pipe, fixed ends)")
print(f"{'='*52}")
print(f"  Max von Mises stress       = {max_vm:.3e} Pa")
print(f"  Max |sigma_rr|             = {max_srr:.3e} Pa")
print(f"  Max displacement magnitude = {max_umag:.3e} m")
print(f"    Analytic u_r(r_i)        = {u_r_exact(r_i):.3e} m")
print(f"    Analytic u_r(r_o)        = {u_r_exact(r_o):.3e} m")
print(f"{'='*52}")