import numpy as np, glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

D = "/home/jonah/dataset4_2026-07-21"

def load(body, stance, obj, site):
    f = "%s/ds4_%s_%s_%s_%s.npz" % (D, body, stance, obj, site)
    return np.load(f, allow_pickle=True) if os.path.exists(f) else None

def panels(ax_m, ax_r, d, title):
    xs, zs = d["xs"], d["zs"]
    ext = [xs[0], xs[-1], zs[0], zs[-1]]
    dm = np.array(d["depth_map"], float)
    dm[~np.isfinite(dm)] = np.nan
    # mesh/body surface: nearer = brighter
    ax_m.imshow(dm.T, origin="lower", extent=ext, cmap="bone_r", aspect="equal")
    ax_m.set_title("mesh surface (depth)\n%s" % title, fontsize=8)
    db = np.array(d["han_db"], float)
    im = ax_r.imshow(db.T, origin="lower", extent=ext, cmap="inferno",
                     vmin=-40, vmax=0, aspect="equal")
    ax_r.set_title("RF reconstruction (dB)", fontsize=8)
    # object box on both
    ab = np.array(d["obj_aabb"], float)
    if np.isfinite(ab).all():
        for ax, c in ((ax_m, "tab:red"), (ax_r, "cyan")):
            ax.add_patch(Rectangle((ab[0,0], ab[0,2]), ab[1,0]-ab[0,0], ab[1,2]-ab[0,2],
                                   fill=False, ec=c, lw=1.4))
    for ax in (ax_m, ax_r):
        ax.set_xticks([]); ax.set_yticks([])
    return im

BODY, STANCE = "m_h50_b50", "back"
SITE = "wb_back_mid"
combos = [("none","nothreat","bare body (baseline)"),
          ("pistol",SITE,"PISTOL @ %s"%SITE),
          ("wallet",SITE,"wallet @ %s"%SITE),
          ("buckle",SITE,"buckle @ %s"%SITE),
          ("name_badge",SITE,"name badge @ %s"%SITE)]
combos = [c for c in combos if load(BODY,STANCE,c[0],c[1]) is not None]
n = len(combos)
fig, axes = plt.subplots(2, n, figsize=(2.5*n, 6.4))
im = None
for j,(o,s,lab) in enumerate(combos):
    d = load(BODY,STANCE,o,s)
    im = panels(axes[0,j], axes[1,j], d, lab)
    c = float(d["han_contrast_db"]) if np.isfinite(d["han_contrast_db"]) else float("nan")
    v = float(d["visibility"]) if np.isfinite(d["visibility"]) else float("nan")
    axes[1,j].set_xlabel("contrast %+.1f dB   vis %.2f" % (c, v), fontsize=7)
fig.suptitle("dataset4 — %s / %s : same body, same site, different objects" % (BODY, STANCE), fontsize=10)
cb = fig.colorbar(im, ax=axes[1,:].tolist(), fraction=0.02, pad=0.01); cb.set_label("dB", fontsize=7)
fig.savefig("/home/jonah/probe/ds4_samples_sameSite.png", dpi=140, bbox_inches="tight")
print("saved sameSite")

# figure 2: pistol at different sites, front stance
BODY2, ST2 = "m_h50_b50", "front"
sites = ["wb_front_mid","belly","thigh_l","wb_left","ankle_l"]
sites = [s for s in sites if load(BODY2,ST2,"pistol",s) is not None]
fig, axes = plt.subplots(2, len(sites), figsize=(2.5*len(sites), 6.4))
for j,s in enumerate(sites):
    d = load(BODY2,ST2,"pistol",s)
    im = panels(axes[0,j], axes[1,j], d, "PISTOL @ %s"%s)
    axes[1,j].set_xlabel("contrast %+.1f dB   vis %.2f" %
                         (float(d["han_contrast_db"]), float(d["visibility"])), fontsize=7)
fig.suptitle("dataset4 — %s / %s : same gun, different carry sites" % (BODY2, ST2), fontsize=10)
cb = fig.colorbar(im, ax=axes[1,:].tolist(), fraction=0.02, pad=0.01); cb.set_label("dB", fontsize=7)
fig.savefig("/home/jonah/probe/ds4_samples_sites.png", dpi=140, bbox_inches="tight")
print("saved sites")
