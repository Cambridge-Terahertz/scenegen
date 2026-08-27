#!/usr/bin/env python3
"""PHASE 0 GATE: is the forward model + backprojection invariant to the coordinate frame?

The migration rests on model_gen being frame-agnostic -- backproject() consumes grid.voxels() as
an (N,3) point set and the kernel only ever forms Euclidean distances, so relabelling the axes
should change nothing physical. This proves or disproves that before any data is regenerated.

Transform tested, current sim frame -> target frame:
    current: (x, y=RANGE, z=vertical)          array in the x-z plane at y=0
    target : (-x, z, y)  ->  x left, y up, z=RANGE into the scene, 0 at the array
    determinant = +1, so right-handed is preserved: a relabelling with one flip, not a mirror.

NOT expected to be bit-identical. The kernel forms dx*dx + dy*dy + dz*dz, and permuting the axes
permutes that SUM; fp32 addition is not associative, so the last ulp may move. The acceptance
criterion is therefore that any difference sits at the fp32 round-off floor (~1e-6 relative), the
same floor the superposition check was validated against (3.2e-6 vs a 6.0e-6 control).
"""
import os, sys, time
import numpy as np

os.environ.setdefault("SMPLX_MODELS", os.path.expanduser("~/smplx_models"))
for p in ("/home/jonah/probe", "/home/jonah/walkthrough",
          "/home/jonah/Documents/code/model_gen/python"):
    if os.path.isdir(p) and p not in sys.path: sys.path.insert(0, p)

import raforward as rf
from raforward import Scene, C0
from recon import ReconGrid, backproject, backproject_points
import body as bodylib
from stance import screening_pose
from mesh_sample import sample_mesh
import trimesh

NFREQ = 128
FREQ = np.linspace(275e9, 305e9, NFREQ)
KV = 2*np.pi*FREQ/C0
STANDOFF = 2.0
NPTS = 200_000
rng = np.random.default_rng(0)

# ---------- geometry, current sim frame ------------------------------------------------
g = np.linspace(-0.067/2, 0.067/2, 130)
xx, zz = np.meshgrid(g, g, indexing="ij")
RX_old = np.column_stack([xx.ravel(), np.zeros(130*130), zz.ravel()])
TX_old = np.array([0.15, 0.0, 0.0])

m = bodylib.get_model("male")
Vp, F = bodylib.posed_verts(m, betas=np.zeros(10), body_pose=screening_pose())
mesh = trimesh.Trimesh(vertices=Vp, faces=F, process=False); mesh.fix_normals()
zc = 0.5*(Vp[:,1].min() + Vp[:,1].max())
bq, bn, _ = sample_mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), C0/290e9/4, front_cull=False)
sel = rng.choice(len(bq), size=min(NPTS, len(bq)), replace=False)
bq = np.asarray(bq)[sel]

def to_sim_old(P):                       # driver's to_sim, front view (yaw 0)
    x, y, z = P[:,0], P[:,1], P[:,2]
    return np.column_stack([x, STANDOFF - z, y - zc])

Q_old = to_sim_old(bq).astype(np.float64)
S = rng.uniform(0.3, 1.0, len(Q_old)).astype(np.float32)
print(f"points {len(Q_old):,}   rx {len(RX_old):,}   nf {NFREQ}", flush=True)
print(f"current frame: x [{Q_old[:,0].min():+.2f},{Q_old[:,0].max():+.2f}]  "
      f"y(range) [{Q_old[:,1].min():+.2f},{Q_old[:,1].max():+.2f}]  "
      f"z(vert) [{Q_old[:,2].min():+.2f},{Q_old[:,2].max():+.2f}]", flush=True)

# ---------- the frame map ---------------------------------------------------------------
def to_target(P):
    """(x, y=range, z=vert) -> (-x, z, y) = (x left, y up, z range)."""
    return np.column_stack([-P[:,0], P[:,2], P[:,1]])

Q_new, TX_new, RX_new = to_target(Q_old), to_target(TX_old[None,:])[0], to_target(RX_old)
M = np.array([[-1,0,0],[0,0,1],[0,1,0]], float)
print(f"transform det = {np.linalg.det(M):+.1f}  (+1 = right-handed preserved)", flush=True)
print(f"target frame : x [{Q_new[:,0].min():+.2f},{Q_new[:,0].max():+.2f}]  "
      f"y(up) [{Q_new[:,1].min():+.2f},{Q_new[:,1].max():+.2f}]  "
      f"z(range) [{Q_new[:,2].min():+.2f},{Q_new[:,2].max():+.2f}]", flush=True)
print(f"array plane check: RX z-spread in target = {np.ptp(RX_new[:,2]):.4f} m (want 0)", flush=True)

def relerr(a, b):
    d = np.abs(a - b); s = np.abs(a) + np.abs(b)
    return float(d.max() / max(np.abs(a).max(), 1e-30)), float(np.median(d[s>0]/s[s>0])*2) if (s>0).any() else 0.0

# ---------- TEST 1: forward model --------------------------------------------------------
t0 = time.time()
sig_old = rf.forward(Scene(q=Q_old, s=S, tx=TX_old[None,:], rx=RX_old, k=KV), precision="fp32")
sig_new = rf.forward(Scene(q=Q_new, s=S, tx=TX_new[None,:], rx=RX_new, k=KV), precision="fp32")
try: sig_old, sig_new = rf.cp.asnumpy(sig_old), rf.cp.asnumpy(sig_new)
except Exception: pass
ident = np.array_equal(sig_old, sig_new)
mx, med = relerr(sig_old, sig_new)
print(f"\nTEST 1 forward   ({time.time()-t0:.0f}s): bit-identical={ident}  "
      f"max rel {mx:.3e}   median rel {med:.3e}", flush=True)

# ---------- TEST 2: dense backprojection -------------------------------------------------
XS = np.arange(-0.45, 0.4501, 0.02); ZS = np.arange(-0.60, 0.6001, 0.02)
YS = np.arange(STANDOFF-0.20, STANDOFF+0.2001, 0.01)
grid_old = ReconGrid(xs=XS, ys=YS, zs=ZS)                 # current: (x, range, vert)
grid_new = ReconGrid(xs=-XS[::-1], ys=ZS, zs=YS)          # target : (x left, up, range)
t0 = time.time()
c_old = backproject(sig_old, tx=TX_old[None,:], rx=RX_old, f=FREQ, grid=grid_old, npad=4096,
                    freq_window="taylor", precision="fp32")
c_new = backproject(sig_new, tx=TX_new[None,:], rx=RX_new, f=FREQ, grid=grid_new, npad=4096,
                    freq_window="taylor", precision="fp32")
try: c_old, c_new = rf.cp.asnumpy(c_old), rf.cp.asnumpy(c_new)
except Exception: pass
# permute the target cube back: (x', y'=vert, z'=range) -> (x, range, vert), undoing the x flip
c_back = np.transpose(c_new, (0, 2, 1))[::-1, :, :]
print(f"  cube shapes old {c_old.shape}  new {c_new.shape}  back {c_back.shape}", flush=True)
ident2 = np.array_equal(c_old, c_back)
mx2, med2 = relerr(np.abs(c_old), np.abs(c_back))
print(f"TEST 2 backproj  ({time.time()-t0:.0f}s): bit-identical={ident2}  "
      f"max rel {mx2:.3e}   median rel {med2:.3e}", flush=True)

# ---------- TEST 3: backproject_points (the envelope path) -------------------------------
pts_old = Q_old[rng.choice(len(Q_old), 20000, replace=False)]
pts_new = to_target(pts_old)
v_old = backproject_points(sig_old, tx=TX_old[None,:], rx=RX_old, f=FREQ, points=pts_old,
                           npad=4096, freq_window="taylor", precision="fp32")
v_new = backproject_points(sig_new, tx=TX_new[None,:], rx=RX_new, f=FREQ, points=pts_new,
                           npad=4096, freq_window="taylor", precision="fp32")
try: v_old, v_new = rf.cp.asnumpy(v_old), rf.cp.asnumpy(v_new)
except Exception: pass
ident3 = np.array_equal(v_old, v_new)
mx3, med3 = relerr(np.abs(v_old), np.abs(v_new))
print(f"TEST 3 bp_points: bit-identical={ident3}  max rel {mx3:.3e}   median rel {med3:.3e}", flush=True)

# ---------- verdict ----------------------------------------------------------------------
FLOOR = 1e-5
worst = max(mx, mx2, mx3)
print(f"\n{'='*64}\nworst max-relative difference: {worst:.3e}   fp32 round-off floor ~{FLOOR:.0e}")
print("VERDICT:", "PASS - physics is frame-invariant, migration is safe"
      if worst <= FLOOR else "FAIL - something in the chain IS axis-aware; investigate before migrating")
print("="*64, flush=True)
