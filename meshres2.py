import os, sys, numpy as np
os.environ.setdefault("SMPLX_MODELS", os.path.expanduser("~/smplx_models"))
os.environ.setdefault("AMASS_ROOT", os.path.expanduser("~/amass"))
for p in (os.path.expanduser("~/walkthrough"), os.path.expanduser("~/Documents/code/model_gen/python")):
    if os.path.isdir(p): sys.path.insert(0, p)
sys.path.insert(0, "/home/jonah/probe")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import trimesh
from stance import screening_pose
import body as bodylib
from mesh_sample import sample_mesh

STANDOFF = 2.0
D = "/home/jonah/dataset4_2026-07-21"
d = np.load(D + "/ds4_m_h50_b50_back_pistol_wb_back_mid.npz", allow_pickle=True)
betas = np.array(d["betas"]); gender = str(d["gender"])
XS = np.array(d["xs"]); ZS = np.array(d["zs"]); stored = np.array(d["depth_map"], float)

model = bodylib.get_model(gender)
Vp, F = bodylib.posed_verts(model, betas=betas, body_pose=screening_pose())
m = trimesh.Trimesh(vertices=Vp, faces=F, process=False); m.fix_normals()
zc = 0.5 * (Vp[:, 1].min() + Vp[:, 1].max())
print("MESH", Vp.shape, F.shape, "area %.3f m2" % m.area)

def to_sim(P, turn, z_center):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    if turn: x, z = -x, -z
    return np.column_stack([x, STANDOFF - z, y - z_center])

q, n, info = sample_mesh(np.asarray(m.vertices), np.asarray(m.faces), 0.0005, front_cull=False)
print("sampled", q.shape, info)
P = to_sim(q.astype(np.float64), True, zc)

def raster(pts, step):
    xs = np.arange(XS[0], XS[-1] + 1e-9, step); zs = np.arange(ZS[0], ZS[-1] + 1e-9, step)
    ix = np.clip(np.round((pts[:,0]-xs[0])/step).astype(np.int32), 0, len(xs)-1)
    iz = np.clip(np.round((pts[:,2]-zs[0])/step).astype(np.int32), 0, len(zs)-1)
    dm = np.full((len(xs), len(zs)), np.inf); np.minimum.at(dm, (ix, iz), pts[:,1])
    dm[np.isinf(dm)] = np.nan
    return dm, xs, zs

fine, fxs, fzs = raster(P, 0.002)
print("fine", fine.shape, "filled %.1f%%" % (100*np.isfinite(fine).mean()))

def shade(dm, ve):
    z = -np.nan_to_num(dm, nan=np.nanmax(dm))
    img = LightSource(azdeg=315, altdeg=45).hillshade(z.T, vert_exag=ve, dx=1, dy=1)
    return np.ma.masked_where(~np.isfinite(dm.T), img)

fig, ax = plt.subplots(1, 3, figsize=(13, 6.8))
ax[0].imshow(shade(stored, 3.0), origin="lower", extent=[XS[0],XS[-1],ZS[0],ZS[-1]],
             cmap="bone", aspect="equal", interpolation="nearest")
ax[0].set_title("STORED in npz: depth_map\n1 cm raster, %d x %d" % stored.shape, fontsize=9)
ax[0].set_xlabel("<- what my earlier figure drew", fontsize=8)
for a in (ax[1], ax[2]):
    a.imshow(shade(fine, 12.0), origin="lower", extent=[fxs[0],fxs[-1],fzs[0],fzs[-1]],
             cmap="bone", aspect="equal", interpolation="nearest")
ax[1].set_title("TRUE SMPL-X mesh, 2 mm raster\n%d x %d" % fine.shape, fontsize=9)
ax[1].set_xlabel("%d verts / %d faces" % (Vp.shape[0], F.shape[0]), fontsize=8)
ax[2].set_title("TRUE mesh - torso zoom", fontsize=9)
ax[2].set_xlim(-0.30, 0.30); ax[2].set_ylim(-0.20, 0.40)
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.suptitle("m_h50_b50 (back) - the npz stores a 1 cm depth raster, NOT the mesh", fontsize=11)
fig.savefig("/home/jonah/probe/ds4_meshres.png", dpi=145, bbox_inches="tight")
print("saved")
