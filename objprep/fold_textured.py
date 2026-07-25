#!/usr/bin/env python
"""Fold the textured folding knife CLOSED: blade = high-y components, rotate about the
hinge so the blade lies down into the handle. Writes pocket_knife_closed.obj + check."""
import sys, numpy as np
sys.path.insert(0, "/Users/jonah/Documents/scratch_results/threat_dataset")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import trimesh, threats
from trimesh.transformations import rotation_matrix
P = lambda a: float(np.max(a) - np.min(a))
SRC = "/Users/jonah/Documents/Blender/Models/threats/knife/knife_textured_7042683/format.obj"
OUT = "/Users/jonah/Documents/Blender/Models/threats/knife/knife_textured_7042683/pocket_knife_closed.obj"
PNG = "/Users/jonah/.claude/jobs/dab08409/tmp/gallery_imgs/_pocketknife_closed.png"
EXTRA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0   # manual tweak on top of auto-align

m = trimesh.load(SRC, process=True, force="mesh")
if isinstance(m, trimesh.Scene): m = trimesh.util.concatenate(tuple(m.geometry.values()))
m.vertices = threats._canon(np.asarray(m.vertices, np.float64))
comps = m.split(only_watertight=False)
blade = [c for c in comps if np.asarray(c.vertices)[:, 1].mean() > 0.30]
handle = [c for c in comps if np.asarray(c.vertices)[:, 1].mean() <= 0.30]
BV = np.vstack([c.vertices for c in blade])
pivot = BV[np.argmin(BV[:, 1])]                      # hinge = lowest blade point
tip = BV[np.argmax(np.linalg.norm(BV[:, :2] - pivot[:2], axis=1))]
v = tip[:2] - pivot[:2]; cur = np.degrees(np.arctan2(v[1], v[0]))
target = 180.0                                       # fold back along -x into the handle
ang = target - cur + EXTRA
print(f"blade {len(blade)} comps, handle {len(handle)}; pivot=({pivot[0]:.2f},{pivot[1]:.2f}) "
      f"tip=({tip[0]:.2f},{tip[1]:.2f}) cur_angle={cur:.0f} -> rotate {ang:.0f}deg")
T = rotation_matrix(np.radians(ang), [0, 0, 1], point=[pivot[0], pivot[1], 0])
out = trimesh.util.concatenate(handle + [c.copy().apply_transform(T) or c for c in
        [cc.copy() for cc in blade]])
# (apply transform properly)
folded = []
for c in blade:
    c = c.copy(); c.apply_transform(T); folded.append(c)
out = trimesh.util.concatenate(handle + folded)
out.vertices = threats._canon(np.asarray(out.vertices)); out.export(OUT)
W = np.asarray(out.vertices)
print(f"closed bbox: {P(W[:,0]):.2f} x {P(W[:,1]):.2f} x {P(W[:,2]):.2f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
Vo = np.asarray(m.vertices); Vo = Vo / P(Vo[:, 0])
ax[0].scatter(Vo[:, 0], Vo[:, 1], s=0.3, c="#9aa6ad", lw=0); ax[0].set_aspect("equal")
ax[0].set_title("OPEN (~50deg, source)", fontsize=10); ax[0].set_xticks([]); ax[0].set_yticks([])
Wn = W / P(W[:, 0])
ax[1].scatter(Wn[:, 0], Wn[:, 1], s=0.3, c="#d94a3d", lw=0); ax[1].set_aspect("equal")
ax[1].set_title(f"CLOSED  {P(W[:,0]):.0f}:{P(W[:,1]):.0f}:{P(W[:,2]):.0f} aspect", fontsize=10)
ax[1].set_xticks([]); ax[1].set_yticks([])
fig.suptitle("Textured pocket knife: open source -> folded closed", fontsize=12)
fig.tight_layout(); fig.savefig(PNG, dpi=120, bbox_inches="tight"); print("wrote", PNG)
