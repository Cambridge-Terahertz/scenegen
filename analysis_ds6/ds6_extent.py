#!/usr/bin/env python3
"""What the EXTENT-only model actually sees.

The extent classifier reaches AUC 0.924 without being told where the object is or what it looks
like -- only its bounding size and voxel count. This draws that box on the reconstruction beside
the true silhouette, then places the same examples in the (width, height) plane the model
partitions, so the 0.924 is legible rather than asserted.

Extent comes from `obj_vox`, the object's occupied cells in the 1 cm recon grid, so widths and
heights below are centimetres directly.
"""
import os, sys, glob
import numpy as np
import sys as _s; _s.path.insert(0,"/home/jonah/probe")
import frame
sys.path.insert(0, "/home/jonah/probe")
exec(open("/home/jonah/probe/ds6_figures.py").read().split("# ---------------------------------"
     "--------------------------------- figure assembly")[0])

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ALL = sorted(glob.glob(os.path.join(D, "**", "*.npz"), recursive=True))
def pick(*frags): return [f for f in ALL if all(v in os.path.basename(f) for v in frags)]
def load(f): return frame.load_scan(f)          # world frame regardless of vintage

m = np.load(os.path.expanduser("~/probe/ds6_meta.npz"), allow_pickle=True)
bb = m["bb"]; nvox = m["nvox"]; y = m["is_threat"].astype(bool); ob = m["obj_name"].astype(str)
ok = np.isfinite(bb[:, 2]) & (nvox > 0)

BODY, VIEW, SITE = "m_h05_b10", "front", "wb_front_mid"
WANT = ["gun1_glock17", "knife_hunting", "phone_modern", "wallet"]
ex = []
for o in WANT:
    c = pick(BODY, f"_{VIEW}_", f"_{o}_", f"_{SITE}.npz")
    if c:
        d = load(c[0])
        ov = np.array(d["obj_vox"])
        # WORLD FRAME: the image plane is (x, y). `zs` is RANGE and obj_vox col 2 is the range
        # index, so the legacy (xs, zs)/col-2 pairing would draw the box at ~1.87 m and report the
        # range depth as a height.
        xs, ys = np.array(d["xs"]), np.array(d["ys"])
        lo, hi = ov.min(0), ov.max(0)
        x0, x1 = xs[lo[0]], xs[hi[0]]; y0, y1 = ys[lo[1]], ys[hi[1]]
        ex.append(dict(d=d, o=o, box=(x0, y0, x1 - x0, y1 - y0),
                       w=(hi[0]-lo[0]+1), h=(hi[1]-lo[1]+1), n=len(ov)))
print("examples:", [e["o"] for e in ex], flush=True)

XL, YL = (-0.33, 0.33), (-0.50, 0.26)
C_T, C_B = "#B4453C", "#1F5F80"
fig = plt.figure(figsize=(12.6, 5.75))
gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0.16,
                      left=0.045, right=0.975, top=0.845, bottom=0.20)
gl = gs[0].subgridspec(1, len(ex), wspace=0.06)

for j, e in enumerate(ex):
    ax = fig.add_subplot(gl[0, j])
    d = e["d"]
    im, _ = rfpan(ax, d)
    img, tag, ext = scene(d, (-0.42, 0.42), (-0.95, 0.95))
    outline(ax, tag, ext, "cyan", 1.1)
    x0, z0, w, h = e["box"]
    ax.add_patch(mpatches.Rectangle((x0, z0), w, h, fill=False, ec="#FFD24A", lw=1.6, ls="-"))
    ax.set_xlim(XL[1], XL[0]); ax.set_ylim(*YL)       # reversed: keeps +x drawn LEFT
    ax.set_xticks([]); ax.set_yticks([])
    thr = THREAT_KIND.get(e["o"])
    ax.set_title(f"{e['o']}\n{'THREAT' if thr else 'benign'}", fontsize=8.5, pad=3,
                 color=C_T if thr else C_B)
    ax.set_xlabel(f"{e['w']} x {e['h']} cm   {e['n']} vox", fontsize=8, labelpad=3)

axs = fig.add_subplot(gs[1])
rng = np.random.default_rng(0)
jit = lambda v: v + rng.normal(0, 0.28, len(v))
for mask, c, lbl in ((ok & ~y, C_B, "benign"), (ok & y, C_T, "threat")):
    axs.scatter(jit(bb[mask, 2]), jit(bb[mask, 3]), s=2.0, c=c, alpha=0.055,
                linewidths=0, rasterized=True)
for e in ex:
    thr = THREAT_KIND.get(e["o"])
    axs.scatter([e["w"]], [e["h"]], s=95, marker="o", facecolor="none",
                edgecolor="#FFD24A", linewidths=2.0, zorder=5)
    axs.annotate(e["o"].replace("gun1_", "").replace("knife_", "").replace("phone_", ""),
                 (e["w"], e["h"]), textcoords="offset points", xytext=(8, 6),
                 fontsize=8, color="#8A6D0B", zorder=6,
                 path_effects=None)
axs.set_xlabel("bounding-box width (cm)", labelpad=2)
axs.set_ylabel("bounding-box height (cm)", labelpad=2)
axs.set_title("the feature space the extent model partitions\n"
              "each dot is one scan; the four examples are ringed", fontsize=9.5, pad=7,
              linespacing=1.5)
axs.grid(alpha=0.2, lw=0.6); axs.set_xlim(0, 40); axs.set_ylim(0, 40)
axs.legend(handles=[Line2D([], [], marker="o", ls="", color=C_T, label="threat", ms=6),
                    Line2D([], [], marker="o", ls="", color=C_B, label="benign", ms=6),
                    Line2D([], [], marker="o", ls="", mfc="none", mec="#FFD24A", mew=2,
                           color="none", label="examples at left", ms=8)],
           loc="upper right", fontsize=8.5, frameon=True, framealpha=0.95)

fig.suptitle("What the extent-only model sees  -  AUC 0.924 from size alone, no location, no pixels",
             fontsize=11.5, y=0.975)
fig.text(0.5, 0.045,
         "Yellow box = the bounding extent fed to the model.  Cyan = the object's true silhouette.  "
         "Size separates most threats from most benigns because guns and knives are simply larger "
         "than wallets and badges -\nbut the classes overlap heavily, which is why extent alone tops "
         "out at 0.924 and the image is still needed for the rest.",
         ha="center", fontsize=8.2, color="0.3")
out = os.path.expanduser("~/probe/ds6_figs/ds6_fig4_extent.png")
fig.savefig(out, dpi=160); print("wrote", out)
