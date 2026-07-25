#!/usr/bin/env python
"""dataset4 figures -- geometry panels rendered from the TRUE full-resolution mesh.

Why this script exists: the first pass drew the geometry from what the npz happens to STORE --
`depth_map` (a 1 cm raster) and `obj_vox` (1 cm voxels). At 1 cm a Glock is ~19x13 cells and a
name badge is 9x5, so the objects came out as blocky red rectangles that say nothing about
shape. Here the body and the object are re-posed and re-seated from their stored parameters and
rasterized at 2 mm, so what you see is the actual geometry that was simulated.

The RF panels still come from the stored `han_db` -- that IS the full-cube render (the driver
does mag.max(1) over the complete reconstruction, not over the gated shell), so there is nothing
finer to regenerate there short of re-running the forward model.

Placement is taken from the npz's own `face` / `bary` / `flip`, so each render is the exact
placement that was simulated, not a re-sample that merely uses the same seed.

Writes into <dataset>/figures/ .
"""
import os, sys, glob
import numpy as np

os.environ.setdefault("SMPLX_MODELS", os.path.expanduser("~/smplx_models"))
os.environ.setdefault("AMASS_ROOT", os.path.expanduser("~/amass"))
for p in (os.path.expanduser("~/walkthrough"),
          os.path.expanduser("~/Documents/code/model_gen/python")):
    if os.path.isdir(p):
        sys.path.insert(0, p)
sys.path.insert(0, "/home/jonah/probe")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
from stance import screening_pose
import body as bodylib, attach, threats

D = os.environ.get("DS4_OUT", "/home/jonah/dataset4_2026-07-21")
FIG = os.path.join(D, "figures")
os.makedirs(FIG, exist_ok=True)
OBJDIR = "/home/jonah/probe/objects_v4"

STANDOFF, ROLL, GAP = 2.0, 0.0, 0.012
NN = 0.0006            # mesh point spacing for the render (vs 0.129 mm for the physics)
STEP = 0.002           # render raster, 2 mm  (the stored raster is 1 cm)
POSE = screening_pose()

STL = next(p for p in glob.glob("/home/jonah/probe/glock17_ext.stl")
           + glob.glob("/home/jonah/**/glock17_ext.stl", recursive=True) if os.path.exists(p))
PHONE = os.path.join(os.path.dirname(STL), "phone_hull.stl")
BENIGN_SPEC = {"phone": ("phone.stl", "alloy"), "wallet": ("wallet.stl", "plastic"),
               "buckle": ("buckle.stl", "steel"), "name_badge": ("name_badge.stl", "polymer")}
OBJECTS = {"pistol": threats.build_library(glock_stl=STL, phone_stl=PHONE)["pistol"]}
for _n, (_f, _mat) in BENIGN_SPEC.items():
    OBJECTS[_n] = threats.Threat(_n, "benign", _mat,
                                 mesh=trimesh.load(f"{OBJDIR}/{_f}", process=False),
                                 conformal=False)

C_BODY = np.array([0.72, 0.78, 0.80])
C_OBJ = np.array([0.95, 0.35, 0.20])
LIGHT = np.array([-0.45, -1.0, 0.55]); LIGHT /= np.linalg.norm(LIGHT)


def load(body, stance, obj, site):
    f = f"{D}/ds4_{body}_{stance}_{obj}_{site}.npz"
    return np.load(f, allow_pickle=True) if os.path.exists(f) else None


def to_sim(P, turn, z_center, is_dir=False):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    if turn:
        x, z = -x, -z
    if is_dir:
        return np.column_stack([x, -z, y])
    return np.column_stack([x, STANDOFF - z, y - z_center])


_BODY_CACHE = {}


def body_of(betas, gender):
    """(model, VC, Vp, F, mesh, z_center, sampled pts/normals) -- cached per body."""
    key = (gender, np.round(np.asarray(betas, float), 6).tobytes())
    if key in _BODY_CACHE:
        return _BODY_CACHE[key]
    model = bodylib.get_model(gender)
    VC = bodylib.canonical_verts(model)
    Vp, F = bodylib.posed_verts(model, betas=np.asarray(betas), body_pose=POSE)
    m = trimesh.Trimesh(vertices=Vp, faces=F, process=False)
    m.fix_normals()
    zc = 0.5 * (Vp[:, 1].min() + Vp[:, 1].max())
    from mesh_sample import sample_mesh
    bq, bn, _ = sample_mesh(np.asarray(m.vertices), np.asarray(m.faces), NN, front_cull=False)
    out = (model, VC, Vp, F, m, zc, bq.astype(np.float64), bn.astype(np.float64))
    _BODY_CACHE[key] = out
    print(f"    built body {gender} ({len(bq):,} render pts)", flush=True)
    return out


def seat(d):
    """Re-seat the stored object using the npz's OWN face/bary/flip. -> (Q, N) in sim frame."""
    name = str(d["obj_name"])
    if name == "none":
        return None, None
    model, VC, Vp, F, m, zc, _, _ = body_of(d["betas"], str(d["gender"]))
    turn = str(d["stance"]) == "back"
    pl = dict(face=int(d["face"]), bary=np.array(d["bary"], float),
              roll=ROLL, standoff=GAP, flip=bool(d["flip"]))
    gq, gn, _ = attach.attach(OBJECTS[name], Vp, VC, F, pl,
                              target_nn=NN, vert_normals=m.vertex_normals)
    return to_sim(gq, turn, zc), to_sim(gn, turn, zc, is_dir=True)


def raster(layers, xlim, zlim, step=STEP):
    """z-buffer a set of (pts, normals, tag) into an RGB image. col0=horiz, col1=depth
    (smaller = nearer the array), col2=vertical."""
    xs = np.arange(xlim[0], xlim[1] + 1e-9, step)
    zs = np.arange(zlim[0], zlim[1] + 1e-9, step)
    dm = np.full((len(xs), len(zs)), np.inf)
    nn = np.zeros((len(xs), len(zs), 3))
    tag = np.zeros((len(xs), len(zs)), np.uint8)
    for pts, nrm, t in layers:
        if pts is None or not len(pts):
            continue
        sel = ((pts[:, 0] >= xs[0]) & (pts[:, 0] <= xs[-1]) &
               (pts[:, 2] >= zs[0]) & (pts[:, 2] <= zs[-1]))
        pts, nrm = pts[sel], nrm[sel]
        if not len(pts):
            continue
        o = np.argsort(-pts[:, 1])                       # farthest first, nearest overwrites
        ix = np.clip(np.round((pts[o, 0] - xs[0]) / step).astype(np.int32), 0, len(xs) - 1)
        iz = np.clip(np.round((pts[o, 2] - zs[0]) / step).astype(np.int32), 0, len(zs) - 1)
        keep = pts[o, 1] < dm[ix, iz]
        ix, iz = ix[keep], iz[keep]
        dm[ix, iz] = pts[o, 1][keep]
        nn[ix, iz] = nrm[o][keep]
        tag[ix, iz] = t
    inten = np.clip(nn @ LIGHT, 0, 1) ** 0.9 * 0.8 + 0.22
    img = np.ones(dm.shape + (3,))
    for t, c in ((1, C_BODY), (2, C_OBJ)):
        img[tag == t] = c * inten[tag == t][:, None]
    return np.clip(img, 0, 1), tag, [xs[0], xs[-1], zs[0], zs[-1]]


def scene(d, xlim, zlim):
    """Full-res render of body + seated object for one scan."""
    _, _, _, _, _, zc, bq, bn = body_of(d["betas"], str(d["gender"]))
    turn = str(d["stance"]) == "back"
    BQ, BN = to_sim(bq, turn, zc), to_sim(bn, turn, zc, is_dir=True)
    GQ, GN = seat(d)
    return raster([(BQ, BN, 1), (GQ, GN, 2)], xlim, zlim)


def show(ax, img, ext, **kw):
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", extent=ext, aspect="equal",
              interpolation="bilinear", **kw)
    ax.set_xticks([]); ax.set_yticks([])


def rf(ax, d, zoom=None):
    xs, zs = np.array(d["xs"]), np.array(d["zs"])
    ext = [xs[0], xs[-1], zs[0], zs[-1]]
    im = ax.imshow(np.array(d["han_db"], float).T, origin="lower", extent=ext, cmap="inferno",
                   vmin=-45, vmax=-10, aspect="equal", interpolation="nearest")
    if zoom:
        ax.set_xlim(*zoom[0]); ax.set_ylim(*zoom[1])
    ax.set_xticks([]); ax.set_yticks([])
    return im


def outline(ax, tag, ext, color="cyan", lw=1.2):
    """TRUE object silhouette (2 mm) as a contour -- replaces the 1 cm voxel rectangle."""
    if not (tag == 2).any():
        return
    ax.contour(np.linspace(ext[0], ext[1], tag.shape[0]),
               np.linspace(ext[2], ext[3], tag.shape[1]),
               (tag == 2).T.astype(float), levels=[0.5], colors=color, linewidths=lw)


def zoom_of(d, pad=0.11):
    ab = np.array(d["obj_aabb"], float)
    cx, cz = 0.5 * (ab[0, 0] + ab[1, 0]), 0.5 * (ab[0, 2] + ab[1, 2])
    h = max(ab[1, 0] - ab[0, 0], ab[1, 2] - ab[0, 2]) * 0.5 + pad
    return (cx - h, cx + h), (cz - h, cz + h)


# ---------------------------------------------------------------- fig 1: objects, one site --
BODY, STANCE, SITE = "m_h50_b50", "back", "wb_back_mid"
combos = [("pistol", SITE, "PISTOL"), ("buckle", SITE, "buckle"),
          ("name_badge", SITE, "name badge"), ("wallet", SITE, "wallet"),
          ("none", "nothreat", "bare body")]
combos = [c for c in combos if load(BODY, STANCE, c[0], c[1]) is not None]
ref = load(BODY, STANCE, "pistol", SITE)
ZX, ZZ = zoom_of(ref)
FX, FZ = (-0.55, 0.55), (-1.0, 1.0)

print("fig 1: objects at one site", flush=True)
fig, ax = plt.subplots(3, len(combos), figsize=(2.7 * len(combos), 9.4))
im = None
for j, (o, s, lab) in enumerate(combos):
    d = load(BODY, STANCE, o, s)
    img, tag, ext = scene(d, FX, FZ)
    show(ax[0, j], img, ext)
    ax[0, j].set_title(lab, fontsize=10, fontweight="bold" if o == "pistol" else "normal")
    zimg, ztag, zext = scene(d, ZX, ZZ)
    show(ax[1, j], zimg, zext)
    im = rf(ax[2, j], d, zoom=(ZX, ZZ))
    outline(ax[2, j], ztag, zext)
    c = float(d["han_contrast_db"]); v = float(d["visibility"])
    ax[2, j].set_xlabel("baseline" if o == "none" else
                        f"contrast {c:+.1f} dB   vis {v:.2f}", fontsize=8)
ax[0, 0].set_ylabel("true mesh, 2 mm render", fontsize=9)
ax[1, 0].set_ylabel("zoom: true geometry", fontsize=9)
ax[2, 0].set_ylabel("zoom: RF reconstruction", fontsize=9)
fig.suptitle(f"dataset4 - {BODY} / {STANCE} @ {SITE}: same body, same site, different objects\n"
             "geometry rendered from the true mesh (2 mm); cyan = true object silhouette",
             fontsize=11)
cb = fig.colorbar(im, ax=ax[2, :].tolist(), fraction=0.02, pad=0.01); cb.set_label("dB", fontsize=8)
fig.savefig(f"{FIG}/fig1_objects_same_site.png", dpi=145, bbox_inches="tight")
plt.close(fig)
print("  saved fig1", flush=True)

# ---------------------------------------------------------------- fig 2: pistol, many sites --
BODY2, ST2 = "m_h50_b50", "front"
sites = [s for s in ("wb_front_mid", "belly", "thigh_l", "wb_left", "ankle_l")
         if load(BODY2, ST2, "pistol", s) is not None]
print("fig 2: pistol carry sites", flush=True)
fig, ax = plt.subplots(2, len(sites), figsize=(2.7 * len(sites), 7.4))
for j, s in enumerate(sites):
    d = load(BODY2, ST2, "pistol", s)
    img, tag, ext = scene(d, FX, FZ)
    show(ax[0, j], img, ext)
    ax[0, j].set_title(s, fontsize=10)
    zx, zz = zoom_of(d)
    zimg, ztag, zext = scene(d, zx, zz)
    im = rf(ax[1, j], d, zoom=(zx, zz))
    outline(ax[1, j], ztag, zext)
    ax[1, j].set_xlabel(f"contrast {float(d['han_contrast_db']):+.1f} dB   "
                        f"vis {float(d['visibility']):.2f}", fontsize=8)
ax[0, 0].set_ylabel("true mesh, 2 mm render", fontsize=9)
ax[1, 0].set_ylabel("zoom: RF reconstruction", fontsize=9)
fig.suptitle(f"dataset4 - {BODY2} / {ST2}: same Glock 17, different carry sites", fontsize=11)
cb = fig.colorbar(im, ax=ax[1, :].tolist(), fraction=0.02, pad=0.01); cb.set_label("dB", fontsize=8)
fig.savefig(f"{FIG}/fig2_pistol_sites.png", dpi=145, bbox_inches="tight")
plt.close(fig)
print("  saved fig2", flush=True)

# ---------------------------------------------------------------- fig 3: why (raster vs true) -
print("fig 3: stored raster vs true geometry", flush=True)
d = ref
stored = np.array(d["depth_map"], float)
xs, zs = np.array(d["xs"]), np.array(d["zs"])
vox = np.array(d["obj_vox"], int)
img, tag, ext = scene(d, FX, FZ)
zimg, ztag, zext = scene(d, ZX, ZZ)
fig, ax = plt.subplots(1, 3, figsize=(14, 6.4))
sd = np.ma.masked_where(~np.isfinite(stored), -np.nan_to_num(stored, nan=0))
ax[0].imshow(sd.T, origin="lower", extent=[xs[0], xs[-1], zs[0], zs[-1]], cmap="bone",
             aspect="equal", interpolation="nearest")
if len(vox):
    ax[0].scatter(xs[vox[:, 0]], zs[vox[:, 2]], s=6, c="red", marker="s", linewidths=0)
ax[0].set_xlim(*ZX); ax[0].set_ylim(*ZZ)
ax[0].set_title("STORED in npz\ndepth_map 1 cm + obj_vox 1 cm", fontsize=10)
ax[0].set_xlabel("what the first figures drew", fontsize=9)
show(ax[1], zimg, zext)
ax[1].set_title("TRUE geometry\n2 mm render of the seated mesh", fontsize=10)
ax[1].set_xlabel("what this figure set draws", fontsize=9)
show(ax[2], img, ext)
ax[2].set_title("full scene", fontsize=10)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"{BODY} / {STANCE} / pistol @ {SITE} - stored 1 cm raster vs true geometry",
             fontsize=11)
fig.savefig(f"{FIG}/fig3_stored_raster_vs_true.png", dpi=145, bbox_inches="tight")
plt.close(fig)
print("  saved fig3", flush=True)

# ------------------------------------------------- fig 5: benign objects, FRONT stance -------
# fig1 is back-stance and fig2 is pistol-only, so neither shows a benign object from the front.
# Two site groups, because no single front site carries all four benign objects: wb_front_mid
# holds the buckle and badge, thigh_l holds the phone and wallet. The pistol repeats in both as
# the same-site threat reference.
print("fig 5: benign objects, front stance", flush=True)
GROUPS = [("wb_front_mid", ["pistol", "buckle", "name_badge"]),
          ("thigh_l", ["pistol", "phone", "wallet"])]
cols = [(s, o) for s, objs in GROUPS for o in objs if load(BODY2, ST2, o, s) is not None]
zooms = {s: zoom_of(load(BODY2, ST2, "pistol", s)) for s, _ in GROUPS}
fig, ax = plt.subplots(3, len(cols), figsize=(2.7 * len(cols), 9.4))
for j, (s, o) in enumerate(cols):
    d = load(BODY2, ST2, o, s)
    img, tag, ext = scene(d, FX, FZ)
    show(ax[0, j], img, ext)
    ax[0, j].set_title(f"{o}\n@ {s}", fontsize=10,
                       fontweight="bold" if o == "pistol" else "normal")
    zx, zz = zooms[s]
    zimg, ztag, zext = scene(d, zx, zz)
    show(ax[1, j], zimg, zext)
    im = rf(ax[2, j], d, zoom=(zx, zz))
    outline(ax[2, j], ztag, zext)
    ax[2, j].set_xlabel(f"contrast {float(d['han_contrast_db']):+.1f} dB   "
                        f"vis {float(d['visibility']):.2f}", fontsize=8)
ax[0, 0].set_ylabel("true mesh, 2 mm render", fontsize=9)
ax[1, 0].set_ylabel("zoom: true geometry", fontsize=9)
ax[2, 0].set_ylabel("zoom: RF reconstruction", fontsize=9)
fig.suptitle(f"dataset4 - {BODY2} / {ST2} stance: benign objects vs the pistol, same site\n"
             "left 3 share wb_front_mid, right 3 share thigh_l; zoom window fixed per group",
             fontsize=11)
cb = fig.colorbar(im, ax=ax[2, :].tolist(), fraction=0.02, pad=0.01); cb.set_label("dB", fontsize=8)
fig.savefig(f"{FIG}/fig5_objects_front_stance.png", dpi=145, bbox_inches="tight")
plt.close(fig)
print("  saved fig5", flush=True)

# ---------------------------------------------------------------- fig 4: object gallery ------
print("fig 4: object gallery", flush=True)
order = ["pistol", "buckle", "phone", "name_badge", "wallet"]
fig, ax = plt.subplots(1, len(order), figsize=(3.0 * len(order), 4.6))
for j, name in enumerate(order):
    t = OBJECTS[name]
    q, n, s = t.sample(0.00025)          # denser than the 0.8 mm raster, so the render fills
    # object frame: long=x, mid=y, thin=z (+z points away from the body, toward the array)
    Q = np.column_stack([q[:, 0], -q[:, 2], q[:, 1]])
    N = np.column_stack([n[:, 0], -n[:, 2], n[:, 1]])
    h = 0.5 * max(np.ptp(q[:, 0]), np.ptp(q[:, 1])) + 0.012
    cx, cz = q[:, 0].mean(), q[:, 1].mean()
    img, tag, ext = raster([(Q, N, 2)], (cx - h, cx + h), (cz - h, cz + h), step=0.0008)
    show(ax[j], img, [e * 100 for e in ext])
    L, W, T = (np.ptp(q[:, i]) * 100 for i in (0, 1, 2))
    ax[j].set_title(f"{name}\n{L:.1f} x {W:.1f} x {T:.1f} cm", fontsize=10,
                    fontweight="bold" if name == "pistol" else "normal")
    # the Glock is a two-material asset (steel slide on a polymer frame); the rest are uniform
    lab = (f"s = {s.min():.2f} - {s.max():.2f}" if np.ptp(s) > 1e-6
           else f"s = {float(np.median(s)):.2f}")
    ax[j].set_xlabel(lab, fontsize=9)
fig.suptitle("dataset4 objects - true mesh, 0.8 mm render, axes in cm (panels not to common scale)",
             fontsize=11)
fig.savefig(f"{FIG}/fig4_object_gallery.png", dpi=145, bbox_inches="tight")
plt.close(fig)
print("  saved fig4", flush=True)

print("DONE ->", FIG, flush=True)
for f in sorted(os.listdir(FIG)):
    print("   ", f, os.path.getsize(os.path.join(FIG, f)) // 1024, "KB")
