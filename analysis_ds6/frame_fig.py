#!/usr/bin/env python3
"""A characteristic scan drawn in the TARGET frame, with axes labelled and signed correctly.

Frame: +z into the scene (0 at the array), +y up, +x LEFT. Right-handed.
The viewer stands at the array looking into the scene, so +x -- the array's left -- is drawn to
the LEFT. Existing npz are in the old frame (x = array's right, z = vertical), so display flips
x and relabels: x_t = -x_old, y_t = z_old, z_t = y_old.

Left panel  : the image plane (x, y) -- what the array sees, range collapsed.
Right panel : a top-down slice (x, z) -- the array at z = 0 at the bottom, scene above, built by
              scattering field_gated back to ABSOLUTE range and max-projecting over height. This
              is the panel that makes the z axis visible and shows the object standing proud.
"""
import os, sys, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

D = "/mnt/shared_drive/Jonah/synthetic_data/dataset6_2026-08-25"
FIG = os.path.expanduser("~/probe/ds6_figs"); os.makedirs(FIG, exist_ok=True)
cand = [f for f in sorted(glob.glob(os.path.join(D,"**","*.npz"), recursive=True))
        if "m_h05_b10" in f and "_front_" in f and "gun1_glock17" in f and "wb_front_mid" in f]
f = cand[0]; z = np.load(f, allow_pickle=True)
print("scan:", os.path.basename(f), flush=True)

xs, ys, zs = np.asarray(z["xs"]), np.asarray(z["ys"]), np.asarray(z["zs"])
han = np.asarray(z["han_db"], float)          # (nx, nz_old) = (x_old, vertical)
fg, gc, dm = z["field_gated"], np.asarray(z["gate_cells"]), np.asarray(z["depth_map"], float)
ov = np.asarray(z["obj_vox"])
nx, nzv = len(xs), len(zs)

# ---- target-frame axes -------------------------------------------------------------------
x_t = -xs                     # +x LEFT
y_t = zs                      # up
z_t = ys                      # range into the scene, 0 at the array

# ---- top-down (x, z): scatter the shell back to ABSOLUTE range, max over height -----------
KG = fg.shape[1]
ix, iv = np.unravel_index(gc, (nx, nzv))
iy0 = np.round((dm.ravel()[gc] - ys[0]) / (ys[1]-ys[0])).astype(np.int32)
off = np.arange(KG) - KG//2
iy = np.clip(iy0[:,None] + off[None,:], 0, len(ys)-1)
top = np.zeros((nx, len(ys)), np.float32)
np.maximum.at(top, (np.repeat(ix, KG), iy.ravel()), np.abs(fg).ravel())
pk = top.max(); top_db = 20*np.log10(np.maximum(top, pk*1e-3)/pk)

# object footprint in the two views
ox_t, oy_t, oz_t = -xs[ov[:,0]], zs[ov[:,2]], ys[ov[:,1]]

plt.rcParams.update({"font.size":9})
fig, ax = plt.subplots(1, 2, figsize=(11.4, 6.3),
                       gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.26))
fig.subplots_adjust(left=0.075, right=0.955, top=0.845, bottom=0.215)

# ---------------- panel A: image plane (x, y) ----------------
a = ax[0]
im = a.pcolormesh(x_t, y_t, han.T, cmap="inferno", vmin=-45, vmax=-10, shading="auto")
a.plot(ox_t, oy_t, ".", ms=1.1, color="#39D6FF", alpha=.55)
a.set_aspect("equal"); a.invert_xaxis()                     # <-- +x to the LEFT
a.set_xlabel("x  (m)   +x to the LEFT  →", labelpad=4)
a.set_ylabel("y  (m)   +y up")
a.set_title("image plane (x, y)\nviewed FROM the array, looking into the scene", fontsize=9.5, pad=7)
a.grid(alpha=.16, lw=.5, color="w")
a.text(0.03, 0.975, "range z collapsed\n(max over 21 bins)", transform=a.transAxes,
       fontsize=7.4, color="w", va="top", ha="left", alpha=.85)

# ---------------- panel B: top-down (x, z) ----------------
b = ax[1]
b.pcolormesh(x_t, z_t, top_db.T, cmap="inferno", vmin=-45, vmax=-10, shading="auto")
b.plot(ox_t, oz_t, ".", ms=1.1, color="#39D6FF", alpha=.55)
b.invert_xaxis()                                            # <-- +x to the LEFT
b.set_xlabel("x  (m)   +x to the LEFT  →", labelpad=4)
b.set_ylabel("z  (m)   +z INTO the scene,  z = 0 at the array")
b.set_title("top-down slice (x, z)\nheight collapsed — shows standoff", fontsize=9.5, pad=7)
b.grid(alpha=.16, lw=.5, color="w")
b.set_ylim(ys.min()-0.03, ys.max()+0.03)
b.annotate("array plane  z = 0  is 1.8 m BELOW this panel\n(+z runs upward, into the scene)",
           xy=(0.03, 0.045), xycoords="axes fraction", fontsize=7.4, color="#9AE6B4",
           va="bottom", ha="left", linespacing=1.5)
b.annotate("", xy=(0.045, 0.30), xytext=(0.045, 0.16), xycoords="axes fraction",
           arrowprops=dict(arrowstyle="-|>", color="#9AE6B4", lw=1.5))

cb = fig.colorbar(im, ax=ax.tolist(), fraction=0.028, pad=0.015)
cb.set_label("dB rel. peak", fontsize=8); cb.ax.tick_params(labelsize=7)

fig.suptitle("Target frame  —  +z into the scene (0 at the array),  +y up,  +x LEFT  "
             "(right-handed)", fontsize=11.5, y=0.955)
fig.text(0.5, 0.038,
         "Cyan = the object's occupied voxels.  Because +x is the array's LEFT and the viewer stands "
         "at the array, +x is drawn to the LEFT: this is a first-person\nview, mirrored relative to "
         "the old plots.  The right panel puts the array at z = 0 with the scene above, so the "
         "2 m standoff and the object's proud\nstandoff both read directly off the z axis.  "
         "Frame matches reflect_array_imaging/processing/bistatic_recon.py.",
         ha="center", fontsize=8.1, color="0.35", linespacing=1.7)
out = os.path.join(FIG, "ds6_fig8_target_frame.png")
fig.savefig(out, dpi=160); print("wrote", out, flush=True)
