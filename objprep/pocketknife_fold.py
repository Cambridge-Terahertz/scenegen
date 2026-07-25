#!/usr/bin/env python
"""Fold the open pocket-knife mesh CLOSED: split blade group from handle group and
rotate the blade ~180 deg about the pivot pin so it tucks back along the handle.
Writes threats/knife/military_pocket_knife/pocket_knife_folded.obj + a check png."""
import sys, numpy as np
sys.path.insert(0, "/Users/jonah/Documents/scratch_results/threat_dataset")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import trimesh, threats
from trimesh.transformations import rotation_matrix
P = lambda a: float(np.max(a) - np.min(a))
SRC = "/Users/jonah/Documents/Blender/Models/threats/knife/military_pocket_knife/Military Pocket Knife.obj"
OUT = "/Users/jonah/Documents/Blender/Models/threats/knife/military_pocket_knife/pocket_knife_folded.obj"
PNG = "/Users/jonah/.claude/jobs/dab08409/tmp/gallery_imgs/_pocketknife_fold.png"
ANGLE = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0

m = trimesh.load(SRC, process=True, force="mesh")
if isinstance(m, trimesh.Scene): m = trimesh.util.concatenate(tuple(m.geometry.values()))
m.vertices = threats._canon(np.asarray(m.vertices, np.float64))    # long=x, mid=y, thin=z
comps = m.split(only_watertight=False)
cx = np.array([c.vertices[:, 0].mean() for c in comps])
split = 0.5 * (cx.min() + cx.max())                               # gap between handle & blade clusters
blade = [c for c in comps if c.vertices[:, 0].mean() > split]
handle = [c for c in comps if c.vertices[:, 0].mean() <= split]
BV = np.vstack([c.vertices for c in blade])
pivot_x = BV[:, 0].min()
pivot_y = BV[BV[:, 0] < pivot_x + 0.05 * P(BV[:, 0])][:, 1].mean()
print(f"{len(comps)} comps -> {len(blade)} blade / {len(handle)} handle; "
      f"blade span x=[{BV[:,0].min():.1f},{BV[:,0].max():.1f}]  pivot=({pivot_x:.1f},{pivot_y:.1f})")

T = rotation_matrix(np.radians(ANGLE), [0, 0, 1], point=[pivot_x, pivot_y, 0])   # swing about the pin
folded = []
for c in blade:
    c = c.copy(); c.apply_transform(T); folded.append(c)
allc = handle + folded
out = trimesh.util.concatenate(allc)
out.vertices = threats._canon(np.asarray(out.vertices))
out.export(OUT)
W = np.asarray(out.vertices)
print(f"folded bbox: {P(W[:,0]):.1f} x {P(W[:,1]):.1f} x {P(W[:,2]):.1f}  (was 58 x 11 x 3 open)")

# check render: open vs folded, profile
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
Vo = np.asarray(m.vertices); Vo = Vo / P(Vo[:, 0])
ax[0].scatter(Vo[:, 0], Vo[:, 1], s=0.6, c="#9aa6ad", lw=0); ax[0].set_aspect("equal")
ax[0].set_title("OPEN (source)", fontsize=10); ax[0].set_xticks([]); ax[0].set_yticks([])
Wn = W / P(W[:, 0])
col = np.where(np.isin(np.arange(len(W)), np.arange(len(W))), "#d94a3d", "#d94a3d")
ax[1].scatter(Wn[:, 0], Wn[:, 1], s=0.6, c="#d94a3d", lw=0); ax[1].set_aspect("equal")
ax[1].set_title(f"FOLDED (blade rot {ANGLE:.0f}deg)  {P(W[:,0]):.0f}x{P(W[:,1]):.0f}x{P(W[:,2]):.1f}", fontsize=10)
ax[1].set_xticks([]); ax[1].set_yticks([])
fig.suptitle("Pocket knife: open source -> procedurally folded closed", fontsize=12)
fig.tight_layout(); fig.savefig(PNG, dpi=120, bbox_inches="tight"); print("wrote", PNG)
