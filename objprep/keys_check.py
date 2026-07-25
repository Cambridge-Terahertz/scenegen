#!/usr/bin/env python
"""Object-only render of the 4 keys variants (and a couple sizes) so the shapes are
unambiguous, independent of the body-seat blur."""
import os, sys, glob
import numpy as np
os.environ["SMPLX_MODELS"] = "/Users/jonah/Documents/code/model_gen/smplx_models"
for p in ("/Users/jonah/Documents/scratch_results/threat_dataset",):
    sys.path.insert(0, p)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import trimesh, threats
_P=lambda a: float(np.max(a)-np.min(a))

MODELS = "/Users/jonah/Documents/Blender/Models"
OUT = "/Users/jonah/.claude/jobs/dab08409/tmp/gallery_imgs"
VARIANTS = ["keys_silver", "keys_golden", "keys_red", "keys_black"]

def load(path, size):
    m = trimesh.load(path, process=True, force="mesh")
    if isinstance(m, trimesh.Scene): m = trimesh.util.concatenate(tuple(m.geometry.values()))
    V = threats._canon(np.asarray(m.vertices, np.float64))
    ext = V[:, 0].max() - V[:, 0].min(); V *= size / ext
    return trimesh.Trimesh(vertices=V, faces=np.asarray(m.faces), process=False)

fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
for col, kv in enumerate(VARIANTS):
    p = glob.glob(os.path.join(MODELS, f"nonthreats/key/variants/{kv}.obj"))[0]
    m = load(p, 0.10)
    V = np.asarray(m.vertices) * 100  # cm
    # top: view along thin axis z -> x (long) horiz vs y (mid) vert
    ax = axes[0, col]
    ax.scatter(V[:, 0], V[:, 1], s=0.6, c="#c33", lw=0)
    ax.set_aspect("equal"); ax.set_title(f"{kv}\nface-on  ({_P(V[:,0]):.0f}x{_P(V[:,1]):.0f} cm)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    # bottom: view along mid axis y -> x horiz vs z (thin) vert  (edge/thickness)
    ax = axes[1, col]
    ax.scatter(V[:, 0], V[:, 2], s=0.6, c="#c33", lw=0)
    ax.set_aspect("equal"); ax.set_title(f"edge-on  (thick {_P(V[:,2]):.1f} cm)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Keys variants at 10 cm long-axis — face-on (top) and edge-on (bottom)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "_keys_variants_check.png"), dpi=110, bbox_inches="tight")
print("wrote _keys_variants_check.png")
