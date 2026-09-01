"""Final acceptance render, read back from the SHIPPED objects_v6 part files (not recomputed),
so what is verified is what the sim will load."""
import os, numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.expanduser("~/gunwork/objects_v6")
GUNS = [("gun1_glock17","Glock 17"),("gun2_glock19","Glock 19"),("gun3_hellcat","FMK 9C1"),
        ("gun4_mp9","Grand Power K100"),("gun5_derringer","Tanfoglio GT27")]
PIX = 0.0010
fig, axes = plt.subplots(1, 5, figsize=(19.5, 4.6))
fig.subplots_adjust(left=.012, right=.99, top=.77, bottom=.15, wspace=.05)
for ax, (n, real) in zip(axes, GUNS):
    C, A, S = [], [], []
    for tag, is_steel in (("steel", 1), ("polymer", 0)):
        f = os.path.join(OUT, f"{n}_{tag}.obj")
        if not os.path.exists(f): continue
        m = trimesh.load(f, process=False)
        C.append(np.asarray(m.triangles_center)); A.append(np.asarray(m.area_faces))
        S.append(np.full(len(m.faces), is_steel))
    C = np.vstack(C); A = np.concatenate(A); S = np.concatenate(S).astype(bool)
    u, v = C[:,0], C[:,1]
    nx = int(np.ptp(u)/PIX)+1; ny = int(np.ptp(v)/PIX)+1
    iu = ((u-u.min())/PIX).astype(int); iv = ((v-v.min())/PIX).astype(int)
    img = np.zeros((ny,nx,2)); np.add.at(img,(iv,iu,S.astype(int)),A)
    tot = img.sum(2); has = tot>0
    st = np.zeros_like(tot); st[has] = img[...,1][has]/tot[has]
    rgb = np.ones((ny,nx,3))
    rgb[has] = np.where(st[has,None]>.5, np.array([.76,.26,.17]), np.array([.05,.44,.72]))
    ax.imshow(rgb, origin="lower")
    a_s, a_p = A[S].sum()*1e4, A[~S].sum()*1e4
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for s in ax.spines.values(): s.set_edgecolor("#D3DAE1")
    ax.set_title(f"{n}\n{real}", fontsize=10, pad=6)
    lbl = ("ALL METAL (cast zinc alloy)" if a_p == 0
           else f"steel {100*a_s/(a_s+a_p):.0f}%   polymer {100*a_p/(a_s+a_p):.0f}%")
    ax.set_xlabel(f"{lbl}\nvisible exterior {a_s+a_p:.0f} cm$^2$", fontsize=8.8,
                  color="#5C6B79", labelpad=6, linespacing=1.6)
fig.suptitle("objects_v6 pistols  —  interiors stripped, materials assigned   "
             "(red = steel slide/barrel,  blue = polymer frame)", fontsize=13, y=.94)
fig.text(0.5, 0.035, "Previously all five were modelled as solid steel. Interior stripping removed "
         "57-65% of the surface of guns 2-5 (bore, magwell, frame internals the sensor can never see).",
         ha="center", fontsize=8.6, color="#5C6B79")
p = os.path.expanduser("~/gunwork/gun_materials.png")
fig.savefig(p, dpi=140, facecolor="white"); print("wrote", p)
