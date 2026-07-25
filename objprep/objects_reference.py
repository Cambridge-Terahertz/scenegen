#!/usr/bin/env python
"""Reference sheet: all confuser objects at TRUE SCALE next to the body, plus a
face-on / edge-on detail with real dimensions. For verifying size + orientation."""
import os, sys, glob
import numpy as np
os.environ["SMPLX_MODELS"] = "/Users/jonah/Documents/code/model_gen/smplx_models"
for p in ("/Users/jonah/Documents/code/model_gen/python",
          "/Users/jonah/Documents/scratch_results/threat_dataset",
          "/Users/jonah/Documents/scratch_results/stop_pose"):
    if os.path.isdir(p): sys.path.insert(0, p)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import trimesh, threats, body as bodylib
_P=lambda a: float(np.max(a)-np.min(a))
from stance import screening_pose
from mesh_sample import sample_mesh

MODELS = "/Users/jonah/Documents/Blender/Models"
OUT = "/Users/jonah/.claude/jobs/dab08409/tmp/gallery_imgs"
C_STEEL, C_POLY = "#d94a3d", "#4a4a8a"
def g(pat):
    m = glob.glob(os.path.join(MODELS, pat)); return m[0] if m else None

# object spec: (path, material, target long-axis m, display: 'vert'|'horiz')
SPEC = {
 "phone":    (g("nonthreats/phone/*nokia*.obj"), "alloy", 0.125, "vert"),
 "wallet":   (g("nonthreats/wallet/*Wallet*.obj"), "plastic", 0.11, "vert"),
 "badge":    (g("nonthreats/badge/variants/name_badge_1.obj"), "polymer", 0.085, "vert"),
 "keys":     (g("nonthreats/key/variants/keys_bunch.obj"), "steel", 0.10, "vert"),
 "necklace": (g("nonthreats/necklace/*.stl"), "steel", 0.05, "vert"),
 "belt":     (g("nonthreats/belt/buckle/Buckle.obj"), "steel", 0.08, "horiz"),
}
MATCOL = {"alloy": C_STEEL, "steel": C_STEEL, "polymer": C_POLY, "plastic": C_POLY}

def load(path, size):
    m = trimesh.load(path, process=True, force="mesh")
    if isinstance(m, trimesh.Scene): m = trimesh.util.concatenate(tuple(m.geometry.values()))
    V = threats._canon(np.asarray(m.vertices, np.float64))
    ext = V[:, 0].max() - V[:, 0].min(); V *= size / ext
    return V  # canonical: long=x, mid=y, thin=z  (metres)

# ---- body silhouette (front), in cm ----
model = bodylib.get_model("neutral")
Vp, F = bodylib.posed_verts(model, betas=np.zeros(10), body_pose=screening_pose())
mbody = trimesh.Trimesh(vertices=Vp, faces=F, process=False); mbody.fix_normals()
BQ, _, _ = sample_mesh(np.asarray(mbody.vertices), np.asarray(mbody.faces), 0.006, front_cull=True)
bx = BQ[:, 0] * 100; bz = (BQ[:, 1] - 0.5*(Vp[:,1].min()+Vp[:,1].max())) * 100   # cm, centered

fig = plt.figure(figsize=(15, 8.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 2.0], wspace=0.14)

# ---- LEFT: body + all objects at TRUE relative scale ----
axb = fig.add_subplot(gs[0, 0])
axb.scatter(bx, bz, s=0.5, c="#9aa6ad", lw=0)
# stack objects in a column to the right of the body, each at true cm scale
oy = np.linspace(75, -95, len(SPEC)); ox = 46
for (name, (path, mat, size, disp)), yc in zip(SPEC.items(), oy):
    V = load(path, size) * 100
    hx, hy = (V[:, 1], V[:, 0]) if disp == "vert" else (V[:, 0], V[:, 1])  # display orient
    axb.scatter(hx - hx.mean() + ox, hy - hy.mean() + yc, s=0.5, c=MATCOL[mat], lw=0)
    axb.text(ox + 16, yc, f"{name}", va="center", fontsize=9, fontweight="bold")
axb.set_aspect("equal"); axb.set_title("True scale vs body (cm)", fontsize=11)
axb.axhline(-105, color="none"); axb.set_xlim(-30, 78); axb.set_ylim(-108, 92)
axb.set_xlabel("cm"); axb.grid(True, alpha=0.15)

# ---- RIGHT: per-object detail, face-on + edge-on, with dimensions ----
gsr = gs[0, 1].subgridspec(2, 3, hspace=0.32, wspace=0.18)
for i, (name, (path, mat, size, disp)) in enumerate(SPEC.items()):
    V = load(path, size) * 100
    ax = fig.add_subplot(gsr[i // 3, i % 3])
    ax.scatter(V[:, 0], V[:, 1], s=0.7, c=MATCOL[mat], lw=0)               # face-on (long x, mid y)
    dz = _P(V[:, 2])
    ax.scatter(V[:, 0], V[:, 2] - _P(V[:, 1])/2 - dz - 2, s=0.7, c=MATCOL[mat], lw=0)  # edge-on below
    ax.set_aspect("equal")
    ax.set_title(f"{name}  ·  {_P(V[:,0]):.0f} × {_P(V[:,1]):.0f} × {dz:.1f} cm",
                 fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color("#ccc")

fig.suptitle("Confuser objects — size & orientation reference  (red = metal, blue = dielectric; "
             "face-on above, edge-on/thickness below)", fontsize=12)
fig.savefig(os.path.join(OUT, "_objects_reference.png"), dpi=115, bbox_inches="tight")
print("wrote _objects_reference.png")
