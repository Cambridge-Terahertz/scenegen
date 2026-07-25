import os, sys, numpy as np
os.environ.setdefault("SMPLX_MODELS", os.path.expanduser("~/smplx_models"))
os.environ.setdefault("AMASS_ROOT", os.path.expanduser("~/amass"))
HERE = "/home/jonah/probe"
for p in (os.path.expanduser("~/walkthrough"), os.path.expanduser("~/Documents/code/model_gen/python")):
    if os.path.isdir(p): sys.path.insert(0, p)
sys.path.insert(0, HERE)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from stance import screening_pose
import body as bodylib
from mesh_sample import sample_mesh

D = "/home/jonah/dataset4_2026-07-21"
d = np.load(D + "/ds4_m_h50_b50_back_pistol_wb_back_mid.npz", allow_pickle=True)
betas = np.array(d["betas"]); gender = str(d["gender"])
print("gender", gender, "betas", np.round(betas, 3))

model = bodylib.get_model(gender)
POSE = screening_pose()
Vp, F = bodylib.posed_verts(model, betas=betas, body_pose=POSE)
print("MESH verts", Vp.shape, "faces", F.shape)

# dense surface sampling of the true mesh
P = sample_mesh(Vp, F, 8_000_000)[0] if isinstance(sample_mesh(Vp, F, 10), tuple) else sample_mesh(Vp, F, 8_000_000)
P = np.asarray(P); print("sampled pts", P.shape)

XS = np.array(d["xs"]); ZS = np.array(d["zs"])
stored = np.array(d["depth_map"], float)

def raster(pts, step):
    xs = np.arange(XS[0], XS[-1] + 1e-9, step); zs = np.arange(ZS[0], ZS[-1] + 1e-9, step)
    ix = np.clip(np.round((pts[:,0]-xs[0])/step).astype(np.int32), 0, len(xs)-1)
    iz = np.clip(np.round((pts[:,2]-zs[0])/step).astype(np.int32), 0, len(zs)-1)
    dm = np.full((len(xs), len(zs)), np.inf)
    np.minimum.at(dm, (ix, iz), pts[:,1])
    dm[np.isinf(dm)] = np.nan
    return dm, xs, zs

fine, fxs, fzs = raster(P, 0.002)
print("fine raster", fine.shape)

def shade(dm):
    z = -np.nan_to_num(dm, nan=np.nanmax(dm))
    ls = LightSource(azdeg=315, altdeg=45)
    img = ls.hillshade(z.T, vert_exag=6.0, dx=1, dy=1)
    img = np.ma.masked_where(~np.isfinite(dm.T), img)
    return img

fig, ax = plt.subplots(1, 3, figsize=(12.5, 6.6))
for a, (dm, xs, zs, t) in zip(ax, [
        (stored, XS, ZS, "STORED in npz: depth_map\n1 cm raster  (%d x %d)" % stored.shape),
        (fine, fxs, fzs, "TRUE SMPL-X mesh, 2 mm raster\n(%d x %d)" % fine.shape),
        (fine, fxs, fzs, "TRUE mesh - zoom on torso")]):
    a.imshow(shade(dm), origin="lower", extent=[xs[0],xs[-1],zs[0],zs[-1]],
             cmap="bone", aspect="equal", interpolation="nearest")
    a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([])
ax[2].set_xlim(-0.25, 0.25); ax[2].set_ylim(-0.15, 0.35)
ax[0].set_xlabel("what my earlier figure rendered", fontsize=8)
ax[1].set_xlabel("%d verts / %d faces" % (Vp.shape[0], F.shape[0]), fontsize=8)
fig.suptitle("dataset4 m_h50_b50 — stored 1 cm depth raster vs the actual SMPL-X mesh", fontsize=11)
fig.savefig("/home/jonah/probe/ds4_meshres.png", dpi=140, bbox_inches="tight")
print("saved ds4_meshres.png")
