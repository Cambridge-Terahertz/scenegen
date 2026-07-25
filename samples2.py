import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = "/home/jonah/dataset4_2026-07-21"
BODY, STANCE = "m_h50_b50", "back"
combos = [("none","nothreat","bare body"),
          ("pistol","wb_back_mid","PISTOL"),
          ("buckle","wb_back_mid","buckle"),
          ("name_badge","wb_back_mid","name badge")]

def L(o,s): return np.load("%s/ds4_%s_%s_%s_%s.npz"%(D,BODY,STANCE,o,s), allow_pickle=True)
d0 = L("pistol","wb_back_mid")
xs, zs = np.array(d0["xs"]), np.array(d0["zs"])
ov = np.array(d0["obj_vox"])
cx, cz = int(ov[:,0].mean()), int(ov[:,2].mean())
P = 26                                    # zoom half-width in cells
x0,x1 = max(cx-P,0), min(cx+P,len(xs)); z0,z1 = max(cz-P,0), min(cz+P,len(zs))
zext = [xs[x0], xs[x1-1], zs[z0], zs[z1-1]]

fig, ax = plt.subplots(3, len(combos), figsize=(3.0*len(combos), 9.6))
for j,(o,s,lab) in enumerate(combos):
    d = L(o,s)
    dm = np.array(d["depth_map"], float); dm[~np.isfinite(dm)] = np.nan
    db = np.array(d["han_db"], float)
    # object footprint: project voxels (max over y) into the (x,z) image plane
    foot = np.zeros_like(dm, bool)
    ovj = np.array(d["obj_vox"])
    if ovj.size: foot[ovj[:,0], ovj[:,2]] = True

    # --- row 0: full body, object painted on ---
    a = ax[0,j]
    a.imshow(dm.T, origin="lower", extent=[xs[0],xs[-1],zs[0],zs[-1]], cmap="bone_r", aspect="equal")
    if foot.any():
        m = np.ma.masked_where(~foot.T, foot.T.astype(float))
        a.imshow(m, origin="lower", extent=[xs[0],xs[-1],zs[0],zs[-1]], cmap="autumn_r",
                 vmin=0, vmax=1, aspect="equal", interpolation="nearest")
    a.plot([xs[x0],xs[x1-1],xs[x1-1],xs[x0],xs[x0]],[zs[z0],zs[z0],zs[z1-1],zs[z1-1],zs[z0]],
           "-", c="tab:blue", lw=0.9)
    a.set_title("%s\nbody mesh + object (red)"%lab, fontsize=9); a.set_xticks([]); a.set_yticks([])

    # --- row 1: ZOOM mesh + object ---
    a = ax[1,j]
    a.imshow(dm[x0:x1, z0:z1].T, origin="lower", extent=zext, cmap="bone_r", aspect="equal")
    if foot[x0:x1, z0:z1].any():
        m = np.ma.masked_where(~foot[x0:x1,z0:z1].T, foot[x0:x1,z0:z1].T.astype(float))
        a.imshow(m, origin="lower", extent=zext, cmap="autumn_r", vmin=0, vmax=1,
                 aspect="equal", interpolation="nearest")
        a.set_xlabel("%d voxels"%ovj.shape[0], fontsize=8)
    else:
        a.set_xlabel("no object", fontsize=8)
    a.set_title("ZOOM: mesh + object shape", fontsize=9); a.set_xticks([]); a.set_yticks([])

    # --- row 2: ZOOM RF ---
    a = ax[2,j]
    im = a.imshow(db[x0:x1, z0:z1].T, origin="lower", extent=zext, cmap="inferno",
                  vmin=-45, vmax=-10, aspect="equal")
    if foot[x0:x1,z0:z1].any():
        a.contour(np.linspace(zext[0],zext[1],x1-x0), np.linspace(zext[2],zext[3],z1-z0),
                  foot[x0:x1,z0:z1].T.astype(float), levels=[0.5], colors="cyan", linewidths=1.2)
    c = float(d["han_contrast_db"]) if np.isfinite(d["han_contrast_db"]) else float("nan")
    a.set_title("ZOOM: RF recon", fontsize=9)
    a.set_xlabel("contrast %+.1f dB"%c if np.isfinite(c) else "baseline", fontsize=8)
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("dataset4 — %s / %s @ wb_back_mid : object geometry vs RF response"%(BODY,STANCE), fontsize=11)
fig.colorbar(im, ax=ax[2,:].tolist(), fraction=0.02, pad=0.01).set_label("dB", fontsize=8)
fig.savefig("/home/jonah/probe/ds4_objects.png", dpi=140, bbox_inches="tight")
print("saved ds4_objects.png")
