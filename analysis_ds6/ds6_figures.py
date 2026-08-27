#!/usr/bin/env python
"""dataset6 figure core -- WORLD FRAME (+z into scene, +y up, +x LEFT).

Geometry panels are rendered from the TRUE mesh at 2 mm; RF panels are the stored han_db, which
IS the full-cube render. Scans are read through frame.load_scan(), so legacy files (ds3-ds5b and
the ds6 pilot) are converted on read and every panel below is in one frame regardless of vintage.

PLOTTING RULE: +x is the array's LEFT and the viewer stands at the array, so every axis is drawn
with invert_xaxis(). Figures are therefore mirrored relative to anything made before 2026-08-26.
"""
import os, sys, glob
import numpy as np

os.environ.setdefault("SMPLX_MODELS", os.path.expanduser("~/smplx_models"))
for p in (os.path.expanduser("~/walkthrough"),
          os.path.expanduser("~/Documents/code/model_gen/python"),
          "/home/jonah/probe"):
    if os.path.isdir(p) and p not in sys.path: sys.path.insert(0, p)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
from stance import screening_pose
import body as bodylib, attach, threats
import placement_config as pc
import frame

D   = os.environ.get("DS6_DIR", "/mnt/shared_drive/Jonah/synthetic_data/dataset6_2026-08-25")
FIG = os.path.expanduser("~/probe/ds6_figs"); os.makedirs(FIG, exist_ok=True)
V5  = next(p for p in (os.path.expanduser("~/probe/objects_v5"),
                       os.path.expanduser("~/walkthrough/objects_v5")) if os.path.isdir(p))
GLOCK = next(p for p in glob.glob("/home/jonah/**/glock17_ext.stl", recursive=True))
PHONE = os.path.join(os.path.dirname(GLOCK), "phone_hull.stl")

STANDOFF = 2.0
NN, STEP = 0.0006, 0.002
POSE = screening_pose()
VIEW_DEG = {"front": 0.0, "back": 180.0, "left": 90.0, "right": -90.0}

OBJECTS = {"gun1_glock17": threats.build_library(glock_stl=GLOCK, phone_stl=PHONE)["pistol"]}
for name, o in pc.OBJECTS.items():
    if name == "gun1_glock17": continue
    OBJECTS[name] = threats.Threat(name, o["kind"], o["material"],
                                   mesh=trimesh.load(f"{V5}/{name}.obj", process=False),
                                   conformal=False)
THREAT_KIND = {n: (o["kind"] == "threat") for n, o in pc.OBJECTS.items()}
THREAT_KIND["gun1_glock17"] = True

C_BODY = np.array([0.72, 0.78, 0.80]); C_OBJ = np.array([0.95, 0.35, 0.20])
# light from the viewer's upper-left; in the world frame -z points back toward the array
LIGHT = np.array([0.45, 0.55, -1.0]); LIGHT /= np.linalg.norm(LIGHT)


def to_world(P, orient, yc, is_dir=False):
    return frame.smplx_to_world(P, VIEW_DEG[orient], yc, STANDOFF, is_dir=is_dir)


_BC = {}
def body_of(betas, gender):
    key = (gender, np.round(np.asarray(betas, float), 6).tobytes())
    if key in _BC: return _BC[key]
    model = bodylib.get_model(gender)
    VC = bodylib.canonical_verts(model)
    Vp, F = bodylib.posed_verts(model, betas=np.asarray(betas), body_pose=POSE)
    m = trimesh.Trimesh(vertices=Vp, faces=F, process=False); m.fix_normals()
    yc = 0.5*(Vp[:,1].min() + Vp[:,1].max())
    from mesh_sample import sample_mesh
    bq, bn, _ = sample_mesh(np.asarray(m.vertices), np.asarray(m.faces), NN, front_cull=False)
    out = (model, VC, Vp, F, m, yc, bq.astype(float), bn.astype(float))
    _BC[key] = out; print(f"    built body {gender} ({len(bq):,} pts)", flush=True)
    return out


def seat(d):
    name = str(d["obj_name"])
    if name == "none": return None, None
    model, VC, Vp, F, m, yc, _, _ = body_of(d["betas"], str(d["gender"]))
    orient = str(d["stance"])
    pl = dict(face=int(d["face"]), bary=np.array(d["bary"], float),
              roll=float(d["roll_deg"]), standoff=float(d["gap_m"]), flip=bool(d["flip"]))
    gq, gn, _ = attach.attach(OBJECTS[name], Vp, VC, F, pl,
                              target_nn=NN, vert_normals=m.vertex_normals)
    return to_world(gq, orient, yc), to_world(gn, orient, yc, is_dir=True)


def raster(layers, xlim, ylim, step=STEP):
    """z-buffer into an RGB image. WORLD frame: col0 = x (left), col1 = y (up), col2 = z (RANGE,
    smaller = nearer the array)."""
    xs = np.arange(xlim[0], xlim[1]+1e-9, step); ys = np.arange(ylim[0], ylim[1]+1e-9, step)
    dm = np.full((len(xs), len(ys)), np.inf); nn = np.zeros((len(xs), len(ys), 3))
    tag = np.zeros((len(xs), len(ys)), np.uint8)
    for pts, nrm, t in layers:
        if pts is None or not len(pts): continue
        sel = ((pts[:,0]>=xs[0])&(pts[:,0]<=xs[-1])&(pts[:,1]>=ys[0])&(pts[:,1]<=ys[-1]))
        pts, nrm = pts[sel], nrm[sel]
        if not len(pts): continue
        o = np.argsort(-pts[:,2])                      # farthest (largest z) first
        ix = np.clip(np.round((pts[o,0]-xs[0])/step).astype(np.int32), 0, len(xs)-1)
        iy = np.clip(np.round((pts[o,1]-ys[0])/step).astype(np.int32), 0, len(ys)-1)
        keep = pts[o,2] < dm[ix,iy]                    # nearer the array wins
        ix, iy = ix[keep], iy[keep]
        dm[ix,iy] = pts[o,2][keep]; nn[ix,iy] = nrm[o][keep]; tag[ix,iy] = t
    inten = np.clip(nn @ LIGHT, 0, 1)**0.9*0.8 + 0.22
    img = np.ones(dm.shape+(3,))
    for t, c in ((1, C_BODY), (2, C_OBJ)): img[tag==t] = c*inten[tag==t][:,None]
    return np.clip(img,0,1), tag, [xs[0], xs[-1], ys[0], ys[-1]]


def scene(d, xlim, ylim):
    _,_,_,_,_, yc, bq, bn = body_of(d["betas"], str(d["gender"]))
    orient = str(d["stance"])
    return raster([(to_world(bq,orient,yc), to_world(bn,orient,yc,True), 1), (*seat(d), 2)],
                  xlim, ylim)


def show(ax, img, ext):
    ax.imshow(np.transpose(img,(1,0,2)), origin="lower", extent=ext, aspect="equal",
              interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([]); ax.invert_xaxis()      # +x drawn to the LEFT


def rfpan(ax, d):
    """han_db is the (x, y) image plane in the world frame."""
    xs, ys = np.asarray(d["xs"]), np.asarray(d["ys"])
    ext = [xs[0], xs[-1], ys[0], ys[-1]]
    im = ax.imshow(np.asarray(d["han_db"], float).T, origin="lower", extent=ext, cmap="inferno",
                   vmin=-45, vmax=-10, aspect="equal", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.invert_xaxis()      # +x drawn to the LEFT
    return im, ext


def outline(ax, tag, ext, color="cyan", lw=1.0):
    if not (tag==2).any(): return
    ax.contour(np.linspace(ext[0],ext[1],tag.shape[0]), np.linspace(ext[2],ext[3],tag.shape[1]),
               (tag==2).T.astype(float), levels=[0.5], colors=color, linewidths=lw)


# ------------------------------------------------------------------ figure assembly ----------
ALL = sorted(glob.glob(os.path.join(D, "**", "*.npz"), recursive=True))
print(f"{len(ALL)} scans", flush=True)
def pick(*frags):
    return [f for f in ALL if all(v in os.path.basename(f) for v in frags)]
def load(f): return frame.load_scan(f)          # world frame regardless of vintage

XL, YL = (-0.42, 0.42), (-0.95, 0.95)
KIND = lambda o: "THREAT" if THREAT_KIND.get(o) else "benign"

def pair(axg, axr, d, title, sub=""):
    img, tag, ext = scene(d, XL, YL)
    show(axg, img, ext); outline(axg, tag, ext, "black", 0.7)
    im, _ = rfpan(axr, d); outline(axr, tag, ext, "cyan", 1.0)
    axr.set_xlim(XL[1], XL[0]); axr.set_ylim(*YL)     # reversed: keeps +x drawn LEFT
    axg.set_title(title, fontsize=8, pad=2)
    if sub: axr.set_xlabel(sub, fontsize=7, labelpad=2)
    return im

def contrast_of(d):
    c = float(np.array(d["han_contrast_db"], float))
    return "no visible points" if not np.isfinite(c) else f"contrast {c:+.1f} dB"

def grid_fig(items, suptitle, fname):
    n = len(items)
    W = max(1.95*n, 7.2)
    fig, ax = plt.subplots(2, n, figsize=(W, 5.9),
                           gridspec_kw=dict(hspace=0.06, wspace=0.04))
    ax = np.atleast_2d(ax)
    if n == 1: ax = ax.reshape(2, 1)
    for j, (d, t) in enumerate(items):
        im = pair(ax[0, j], ax[1, j], d, t, contrast_of(d))
    ax[0, 0].set_ylabel("geometry (2 mm mesh)", fontsize=8)
    ax[1, 0].set_ylabel("RF reconstruction", fontsize=8)
    fig.suptitle(suptitle, fontsize=9.5, y=0.975, va="top")
    fig.subplots_adjust(left=0.06, right=0.9, top=0.845, bottom=0.05)
    cax = fig.add_axes([0.915, 0.06, 0.014, 0.38])
    cb = fig.colorbar(im, cax=cax); cb.set_label("dB rel. peak", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(f"{FIG}/{fname}", dpi=155); plt.close(fig)
    print("wrote", fname, flush=True)

BODY = "m_h05_b10"

# ---- FIG 1: same body, same site -- threat vs benign -------------------------------------
VIEW, SITE = "front", "wb_front_mid"
items = []
for o in ["gun1_glock17", "knife_hunting", "phone_modern", "wallet"]:
    c = pick(BODY, f"_{VIEW}_", f"_{o}_", f"_{SITE}.npz")
    if c:
        items.append((load(c[0]), f"{o}\n{KIND(o)}"))
if items:
    grid_fig(items,
             f"dataset6 - same body, same site ({SITE}, {VIEW} view)\n"
             "identical placement, different object: the label is object identity, not location",
             "ds6_fig1_threat_vs_benign.png")

# ---- FIG 2: ONE object+site across every view that sees it -------------------------------
O2, S2 = "gun1_glock17", "wb_r"
items = []
for v in ["front", "right", "back", "left"]:
    c = pick(BODY, f"_{v}_", f"_{O2}_", f"_{S2}.npz")
    if c:
        d = load(c[0])
        roll = float(np.array(d["roll_deg"], float))
        gap = float(np.array(d["gap_m"], float)) * 1000
        items.append((d, f"{v} view\nroll {roll:+.1f} deg   gap {gap:.1f} mm"))
if items:
    grid_fig(items,
             f"dataset6 - {O2} at {S2}, from every view that sees the site\n"
             "roll and gap differ per view: the view-seed bug makes these four scenes, not four aspects",
             "ds6_fig2_four_views.png")

# ---- FIG 3: the visibility tail -- barely-visible vs comfortably visible -----------------
# True zero-visibility scans record NO face/bary (the driver only writes placement provenance
# inside the `if ng:` branch), so they cannot be re-seated for a render. The nearest honest
# illustration is the low tail: the same object and site imaged from an aspect that barely sees
# it. The geometry panel z-buffers from the sensor viewpoint, so the occlusion is visible as
# occlusion rather than asserted in a caption.
m = np.load(os.path.expanduser("~/probe/ds6_meta.npz"), allow_pickle=True)
vis = m["visibility"].astype(float); ob = m["obj_name"].astype(str)
thr = np.array([THREAT_KIND.get(o, False) for o in ob])
ok = np.isfinite(vis) & (vis > 0) & thr
order = np.argsort(np.where(ok, vis, np.inf))
items = []
lo = int(order[0])
dz = load(ALL[lo]); o, s, v = str(dz["obj_name"]), str(dz["site"]), str(dz["stance"])
items.append((dz, f"{o} at {s}\n{v} view - vis {float(vis[lo]):.3f}"))
for w in [x for x in ["front", "back", "left", "right"] if x != v]:
    c = pick(str(dz["body"]), f"_{w}_", f"_{o}_", f"_{s}.npz")
    if c:
        d2 = load(c[0])
        vv = float(np.array(d2["visibility"], float))
        if np.isfinite(vv) and vv > 0.2:
            items.append((d2, f"same object + site\n{w} view - vis {vv:.3f}")); break
if items:
    grid_fig(items,
             "dataset6 - the visibility tail, and why the relabel was needed\n"
             "1,559 scans (4.9%) see the object not at all; 842 of those were labelled threat",
             "ds6_fig3_visibility_tail.png")
