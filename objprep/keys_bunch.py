#!/usr/bin/env python
"""Make a 3D 'bunch' keyring from the flat keys_black layout: split into components,
splay each key about the hang (long) axis so they fan OUT of plane into a thick wad.
Writes nonthreats/key/variants/keys_bunch.obj and a 3-view check png."""
import os, sys, numpy as np
sys.path.insert(0, "/Users/jonah/Documents/scratch_results/threat_dataset")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import trimesh, threats
from trimesh.transformations import rotation_matrix
P = lambda a: float(np.max(a) - np.min(a))

SRC = "/Users/jonah/Documents/Blender/Models/nonthreats/key/variants/keys_black.obj"
OUTOBJ = "/Users/jonah/Documents/Blender/Models/nonthreats/key/variants/keys_bunch.obj"
OUTPNG = "/Users/jonah/.claude/jobs/dab08409/tmp/gallery_imgs/_keys_bunch_check.png"

m = trimesh.load(SRC, process=True, force="mesh")
V = threats._canon(np.asarray(m.vertices, np.float64))          # long=x, mid=y, thin=z
m = trimesh.Trimesh(vertices=V, faces=np.asarray(m.faces), process=True)
comps = m.split(only_watertight=False)
comps = sorted(comps, key=lambda c: -c.vertices[:, 1].mean())   # by mid-axis position
print(f"{len(comps)} components")

# splay each about the x (hang) axis; central pieces (rings) move least.
rng = np.random.default_rng(1)
angles = np.linspace(-70, 70, len(comps))
rng.shuffle(angles)
out = []
for c, ang in zip(comps, angles):
    T = rotation_matrix(np.radians(ang), [1, 0, 0], point=[c.vertices[:, 0].max(), 0, 0])
    # small random tilt about mid-axis so they don't stay perfectly radial
    T2 = rotation_matrix(np.radians(rng.uniform(-12, 12)), [0, 1, 0], point=c.vertices.mean(0))
    c = c.copy(); c.apply_transform(T2 @ T); out.append(c)
bunch = trimesh.util.concatenate(out)
bunch.vertices = threats._canon(np.asarray(bunch.vertices))     # re-canon so long=x again
bunch.export(OUTOBJ)
W = np.asarray(bunch.vertices); W = W / (W[:, 0].max() - W[:, 0].min()) * 0.10 * 100  # 10cm, cm
print(f"bunch bbox: {P(W[:,0]):.1f} x {P(W[:,1]):.1f} x {P(W[:,2]):.1f} cm  "
      f"thin/long={P(W[:,2])/P(W[:,0]):.2f}  (flat<0.1, bunched>0.25)")

idx = rng.choice(len(W), min(50000, len(W)), replace=False); Ws = W[idx]
fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
for a, (i, j, t) in zip(ax, [(0, 1, "face-on"), (0, 2, "edge-on (thickness!)"), (1, 2, "end-on")]):
    a.scatter(Ws[:, i], Ws[:, j], s=0.4, c="#c33", lw=0); a.set_aspect("equal")
    a.set_title(t, fontsize=10); a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"keys_bunch (splayed 3D)  {P(W[:,0]):.0f} x {P(W[:,1]):.0f} x {P(W[:,2]):.1f} cm  "
             f"thin/long={P(W[:,2])/P(W[:,0]):.2f}", fontsize=12)
fig.tight_layout(); fig.savefig(OUTPNG, dpi=115, bbox_inches="tight")
print("wrote", OUTOBJ, "and check png")
