#!/usr/bin/env python3
"""The clean-body reference: what a no-threat scan looks like, and what an object adds to it.

Every body contributes 4 baseline scans (one per view) carrying no object at all. They are the
anchor for reading every other panel: the bright bands, the hand and shoulder glints and the
speckled torso are all BODY, not threat. Fig 5 shows the bare body from each view; fig 6 puts a
clean body beside the same body wearing a Glock and takes the difference, which isolates the
object's own contribution from the body it sits on.
"""
import os, sys, glob
import numpy as np
import sys as _s; _s.path.insert(0,"/home/jonah/probe")
import frame
sys.path.insert(0, "/home/jonah/probe")
exec(open("/home/jonah/probe/ds6_figures.py").read().split("# ---------------------------------"
     "--------------------------------- figure assembly")[0])

ALL = sorted(glob.glob(os.path.join(D, "**", "*.npz"), recursive=True))
def pick(*frags): return [f for f in ALL if all(v in os.path.basename(f) for v in frags)]
def load(f): return frame.load_scan(f)          # world frame regardless of vintage

BODY = "m_h05_b10"
XL, YL = (-0.42, 0.42), (-0.95, 0.95)

def panel(axg, axr, d, title, sub):
    img, tag, ext = scene(d, XL, YL)
    show(axg, img, ext)
    if (tag == 2).any(): outline(axg, tag, ext, "black", 0.7)
    im, _ = rfpan(axr, d)
    if (tag == 2).any(): outline(axr, tag, ext, "cyan", 1.0)
    axr.set_xlim(XL[1], XL[0]); axr.set_ylim(*YL)     # reversed: keeps +x drawn LEFT
    axg.set_title(title, fontsize=8.5, pad=3)
    axr.set_xlabel(sub, fontsize=8, labelpad=3)
    return im

# ---------------- FIG 5: bare body, all four views -------------------------------------
items = []
for v in ["front", "right", "back", "left"]:
    c = pick(BODY, f"_{v}_", "_none_nothreat.npz")
    if c:
        d = load(c[0])
        pk = float(np.array(d["han_peak"], float))
        items.append((d, f"{v} view", "no object" if not items else ""))
print("clean views:", len(items), flush=True)
n = len(items)
fig, ax = plt.subplots(2, n, figsize=(max(1.95*n, 7.2), 5.9),
                       gridspec_kw=dict(hspace=0.06, wspace=0.04))
ax = np.atleast_2d(ax)
for j, (d, t, s) in enumerate(items):
    im = panel(ax[0, j], ax[1, j], d, t, s)
ax[0, 0].set_ylabel("geometry (2 mm mesh)", fontsize=8)
ax[1, 0].set_ylabel("RF reconstruction", fontsize=8)
fig.suptitle("dataset6 - the clean body, every view\n"
             "bands, hand and shoulder glints and speckled torso are all BODY:\n"
             "this is what a threat has to be distinguished FROM",
             fontsize=9.5, y=0.985, va="top")
fig.subplots_adjust(left=0.06, right=0.9, top=0.80, bottom=0.05)
cax = fig.add_axes([0.915, 0.06, 0.014, 0.38])
cb = fig.colorbar(im, cax=cax); cb.set_label("dB rel. peak", fontsize=7); cb.ax.tick_params(labelsize=6)
fig.savefig(f"{FIG}/ds6_fig5_clean_body.png", dpi=160); plt.close(fig)
print("wrote ds6_fig5_clean_body.png", flush=True)

# ---------------- FIG 6: clean vs object vs difference ----------------------------------
VIEW, SITE, OBJ = "front", "wb_front_mid", "gun1_glock17"
cf = pick(BODY, f"_{VIEW}_", "_none_nothreat.npz")
gf = pick(BODY, f"_{VIEW}_", f"_{OBJ}_", f"_{SITE}.npz")
if cf and gf:
    dc, dg = load(cf[0]), load(gf[0])
    fig, ax = plt.subplots(2, 3, figsize=(8.4, 5.9),
                           gridspec_kw=dict(hspace=0.06, wspace=0.05))
    im = panel(ax[0, 0], ax[1, 0], dc, "clean body", "no object")
    im = panel(ax[0, 1], ax[1, 1], dg, f"{OBJ}\nat {SITE}",
               f"contrast {float(np.array(dg['han_contrast_db'],float)):+.1f} dB")
    # difference of the two dB images -- what the object ADDS, on the body it sits on
    a = np.array(dg["han_db"], float); b = np.array(dc["han_db"], float)
    # WORLD FRAME: han_db is the (x, y) image plane. `zs` is RANGE here, so taking the vertical
    # from it -- as the legacy code did -- puts the panel at ~1.87 m, far outside the body.
    xs, ys = np.array(dg["xs"]), np.array(dg["ys"])
    ext = [xs[0], xs[-1], ys[0], ys[-1]]
    img, tag, gext = scene(dg, XL, YL)
    ov = np.array(dg["obj_vox"]); lo, hi = ov.min(0), ov.max(0)
    PAD = 0.13
    zx = (xs[lo[0]] - PAD, xs[hi[0]] + PAD)      # obj_vox cols are (x, y, z=range)
    zy = (ys[lo[1]] - PAD, ys[hi[1]] + PAD)
    show(ax[0, 2], img, gext); outline(ax[0, 2], tag, gext, "black", 0.9)
    ax[0, 2].set_xlim(zx[1], zx[0]); ax[0, 2].set_ylim(*zy)   # reversed: +x LEFT
    ax[0, 2].set_title("object region, zoomed", fontsize=8.5, pad=3)
    dm = ax[1, 2].imshow((a - b).T, origin="lower", extent=ext, cmap="RdBu_r",
                         vmin=-12, vmax=12, aspect="equal", interpolation="nearest")
    outline(ax[1, 2], tag, gext, "black", 1.2)
    ax[1, 2].set_xlim(zx[1], zx[0]); ax[1, 2].set_ylim(*zy)   # reversed: +x LEFT
    ax[1, 2].set_xticks([]); ax[1, 2].set_yticks([])
    ax[1, 2].set_xlabel("difference, same zoom", fontsize=8, labelpad=3)
    ax[0, 0].set_ylabel("geometry (2 mm mesh)", fontsize=8)
    ax[1, 0].set_ylabel("RF reconstruction", fontsize=8)
    fig.suptitle("dataset6 - what the object actually adds\n"
                 "differencing against the same body's clean scan isolates the object from the body",
                 fontsize=9.5, y=0.975, va="top")
    fig.subplots_adjust(left=0.075, right=0.87, top=0.845, bottom=0.05)
    c1 = fig.add_axes([0.885, 0.52, 0.016, 0.26]); cb1 = fig.colorbar(im, cax=c1)
    cb1.set_label("dB rel. peak", fontsize=7); cb1.ax.tick_params(labelsize=6)
    c2 = fig.add_axes([0.885, 0.10, 0.016, 0.26]); cb2 = fig.colorbar(dm, cax=c2)
    cb2.set_label("dB change", fontsize=7); cb2.ax.tick_params(labelsize=6)
    fig.savefig(f"{FIG}/ds6_fig6_object_delta.png", dpi=160); plt.close(fig)
    print("wrote ds6_fig6_object_delta.png", flush=True)
