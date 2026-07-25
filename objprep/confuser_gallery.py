#!/usr/bin/env python
"""Render each confuser object seated at every candidate body site (geometry only,
no forward model). Produces one PNG per (object, site) = full-body + zoom, plus a
manifest.json. Feeds the annotation Artifact.

SMOKE=1 -> only 'keys' at 3 sites, to validate seating/scale/orientation first.
"""
import os, sys, glob, json
import numpy as np

# --- real SMPL-X models (NOT ~/smplx_models, which is empty) ---
os.environ["SMPLX_MODELS"] = "/Users/jonah/Documents/code/model_gen/smplx_models"
for p in ("/Users/jonah/Documents/code/model_gen/python",
          "/Users/jonah/Documents/scratch_results/threat_dataset",
          "/Users/jonah/Documents/scratch_results/stop_pose"):
    if os.path.isdir(p):
        sys.path.insert(0, p)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
from stance import screening_pose
import body as bodylib, attach, threats, anatomy
from mesh_sample import sample_mesh

MODELS = "/Users/jonah/Documents/Blender/Models"
OUT = "/Users/jonah/.claude/jobs/dab08409/tmp/gallery_imgs"
os.makedirs(OUT, exist_ok=True)
STANDOFF, GAP, ROLL = 2.0, 0.012, 0.0
NN_BODY, NN_OBJ = 0.0035, 0.0012
POSE = screening_pose()
SMOKE = os.environ.get("SMOKE") == "1"

C_BODY = np.array([0.72, 0.78, 0.80])
C_STEEL = np.array([0.851, 0.290, 0.239]); C_POLY = np.array([0.290, 0.290, 0.541])
LIGHT = np.array([-0.45, -1.0, 0.55]); LIGHT /= np.linalg.norm(LIGHT)

# ---- inject the 5 sheet sites missing from anatomy.SITES (mirror the ankle azimuth rule:
#      +90 = normal toward model's LEFT/+x, -90 = toward model's RIGHT/-x) ----
anatomy.SITES.update({
    "thigh_inner_l": ("left_thigh",  -90.0, None, None),   # left leg inner faces -x
    "thigh_inner_r": ("right_thigh", +90.0, None, None),
    "thigh_outer_l": ("left_thigh",  +90.0, None, None),
    "thigh_outer_r": ("right_thigh", -90.0, None, None),
    "upper_arm_l":   ("left_upper_arm",  "any", None, None),
    "forearm_r":     ("right_forearm",   "any", None, None),
})
# sheet-name -> anatomy-name (naming differences only)
ALIAS = {"wb_front_l": "wb_front_left", "wb_front_r": "wb_front_right",
         "wb_back_l": "wb_back_left", "wb_back_r": "wb_back_right",
         "wb_l": "wb_left", "wb_r": "wb_right"}
SHEET_SITES = ["wb_front_mid","wb_front_l","wb_front_r","flank_l","flank_r",
               "wb_back_mid","wb_back_l","wb_back_r","wb_l","wb_r",
               "belly","chest","small_of_back","upper_back",
               "thigh_inner_l","thigh_inner_r","thigh_outer_l","thigh_outer_r",
               "ankle_l","ankle_r","upper_arm_l","upper_arm_r","forearm_l","forearm_r",
               "shin_l","shin_r"]
BACK_SITES = {"small_of_back","upper_back","wb_back_mid","wb_back_l","wb_back_r"}

# ---- confuser objects: mesh path, material, target long-axis size (m), roll offset (deg) ----
# roll offsets come from the reviewer's notes (constant per object across all sites):
#   badge/keys/phone upside-down -> +180 ; belt -> +90 (buckle sits horizontal)
def g(pat):
    m = glob.glob(os.path.join(MODELS, pat));
    return m[0] if m else None
CONFUSERS = {
    "phone":    dict(path=g("nonthreats/phone/*nokia*.obj"), mat="alloy",   size=0.125, roll=180),
    "wallet":   dict(path=g("nonthreats/wallet/*Wallet*.obj"), mat="plastic", size=0.11,  roll=0),
    "badge":    dict(path=g("nonthreats/badge/variants/name_badge_1.obj"), mat="polymer", size=0.085, roll=180),
    "keys":     dict(path=g("nonthreats/key/variants/keys_bunch.obj"), mat="steel", size=0.10,  roll=180),
    "necklace": dict(path=g("nonthreats/necklace/*.stl"),   mat="steel",   size=0.05,  roll=0),
    "belt":     dict(path=g("nonthreats/belt/buckle/Buckle.obj"), mat="steel", size=0.08, roll=90),
}
PHONE_VARIANTS = {  # for the one-off scale/shape comparison (TEST mode)
    "iphone": g("nonthreats/phone/*iphone*.obj"),
    "nokia":  g("nonthreats/phone/*nokia*.obj"),
    "mobiele": g("nonthreats/phone/mobiele*.obj"),
}

def load_confuser(path, material, target_long):
    """Load, canonicalize (long=x,mid=y,thin=z), scale long axis to target metres."""
    m = trimesh.load(path, process=True, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    V = threats._canon(np.asarray(m.vertices, np.float64))     # PCA-orient + center
    ext = V[:, 0].max() - V[:, 0].min()
    V *= (target_long / ext) if ext > 0 else 1.0
    m2 = trimesh.Trimesh(vertices=V, faces=np.asarray(m.faces), process=False)
    m2.fix_normals()
    return threats.Threat("confuser", "benign", material, mesh=m2, conformal=False)

def to_sim(P, turn, zc, is_dir=False):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    if turn: x, z = -x, -z
    if is_dir: return np.column_stack([x, -z, y])
    return np.column_stack([x, STANDOFF - z, y - z_center_holder[0]])

# body (neutral, screening pose), cached
print("building body ...", flush=True)
model = bodylib.get_model("neutral")
VC = bodylib.canonical_verts(model)
Vp, F = bodylib.posed_verts(model, betas=np.zeros(10), body_pose=POSE)
mbody = trimesh.Trimesh(vertices=Vp, faces=F, process=False); mbody.fix_normals()
z_center = 0.5 * (Vp[:, 1].min() + Vp[:, 1].max())
z_center_holder = [z_center]
BQ, BN, _ = sample_mesh(np.asarray(mbody.vertices), np.asarray(mbody.faces), NN_BODY, front_cull=False)
BQ = BQ.astype(np.float64); BN = BN.astype(np.float64)
print(f"  body {len(BQ):,} pts, z_center={z_center:.3f}", flush=True)

def seat(threat, sheet_site, roll=0.0):
    aname = ALIAS.get(sheet_site, sheet_site)
    pl = anatomy.sample_placement(model, aname, np.random.default_rng(0), jitter=False)
    pl = dict(face=pl["face"], bary=np.array(pl["bary"], float),
              roll=ROLL + roll, standoff=GAP, flip=False)
    q, n, s = attach.attach(threat, Vp, VC, F, pl, target_nn=NN_OBJ,
                            vert_normals=mbody.vertex_normals)
    return np.asarray(q), np.asarray(n), np.asarray(s, float)

def shade(normals):
    return 0.35 + 0.65 * np.clip(normals @ LIGHT, 0, 1)

def draw(ax, turn, oq, on, os_, zoom=None):
    # z-buffer body + object onto (x horiz, z vert); nearer array (smaller y) wins
    bq = to_sim(BQ, turn, z_center); bn = to_sim(BN, turn, z_center, is_dir=True)
    oqs = to_sim(oq, turn, z_center); ons = to_sim(on, turn, z_center, is_dir=True)
    step = 0.006 if zoom is None else 0.0025
    if zoom is None: xlim, zlim = (-0.55, 0.55), (-1.05, 1.05)
    else:
        cx, cz = oqs[:, 0].mean(), oqs[:, 2].mean()
        xlim, zlim = (cx - 0.18, cx + 0.18), (cz - 0.22, cz + 0.22)
    xs = np.arange(xlim[0], xlim[1], step); zs = np.arange(zlim[0], zlim[1], step)
    nx, nz = len(xs), len(zs)
    img = np.ones((nz, nx, 3)); depth = np.full((nz, nx), 1e9)
    def splat(q, n, colors, rad=0):
        ix = np.round((q[:, 0] - xlim[0]) / step).astype(int)
        iz = np.round((q[:, 2] - zlim[0]) / step).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
        ix, iz, yy = ix[ok], iz[ok], q[ok, 1]; col = colors[ok]
        order = np.argsort(-yy)                      # far first, so near overwrites
        for j in order:
            for dz in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    a, b = iz[j] + dz, ix[j] + dx
                    if 0 <= a < nz and 0 <= b < nx and yy[j] <= depth[a, b] + 1e-4:
                        depth[a, b] = min(depth[a, b], yy[j]); img[a, b] = col[j]
    splat(bq, bn, C_BODY[None] * shade(bn)[:, None])
    ocol = np.where((os_ > 0.5)[:, None], C_STEEL[None], C_POLY[None]) * shade(ons)[:, None]
    splat(oqs, ons, ocol, rad=1 if zoom is None else 2)   # dilate object so it reads
    ax.imshow(img, origin="lower", extent=[xlim[0], xlim[1], zlim[0], zlim[1]], aspect="equal")
    if zoom is None:                                  # locator ring on the full-body view
        cx, cz = oqs[:, 0].mean(), oqs[:, 2].mean()
        ax.add_patch(plt.Circle((cx, cz), 0.09, fill=False, color="#e8a800", lw=1.3))
    ax.set_xticks([]); ax.set_yticks([])

def render(obj_name, threat, sheet_site, roll=0.0, tag=""):
    turn = sheet_site in BACK_SITES
    try:
        oq, on, os_ = seat(threat, sheet_site, roll)
    except Exception as e:
        print(f"  SEAT FAIL {obj_name}@{sheet_site}: {e}", flush=True); return None
    fig, ax = plt.subplots(1, 2, figsize=(4.2, 3.2))
    draw(ax[0], turn, oq, on, os_, zoom=None)
    draw(ax[1], turn, oq, on, os_, zoom=True)
    steel_frac = float((os_ > 0.5).mean())
    ax[0].set_title(f"{obj_name} @ {sheet_site}", fontsize=8)
    ax[1].set_title(f"zoom · {'back' if turn else 'front'} · metal {steel_frac:.0%}", fontsize=7)
    fig.tight_layout(pad=0.3)
    fn = f"{obj_name}{('_'+tag) if tag else ''}__{sheet_site}.png"
    fig.savefig(os.path.join(OUT, fn), dpi=100, bbox_inches="tight"); plt.close(fig)
    return fn

# ---- run ----
if os.environ.get("TEST") == "1":
    # orientation check: each object at 2 sites, + 3 phone variants at wb_front_mid
    for oname, sp in CONFUSERS.items():
        th = load_confuser(sp["path"], sp["mat"], sp["size"])
        for s in ("wb_front_mid", "thigh_outer_r"):
            render(oname, th, s, sp["roll"]); print(f"  {oname} @ {s} (roll {sp['roll']})", flush=True)
    for vname, vpath in PHONE_VARIANTS.items():
        if vpath:
            th = load_confuser(vpath, "alloy", 0.125)
            render("phonevar", th, "wb_front_mid", 180, tag=vname); print(f"  phone variant {vname}", flush=True)
    print("TEST renders done", flush=True); sys.exit()

if SMOKE:
    objs = {"keys": CONFUSERS["keys"]}; sites = ["belly", "thigh_outer_r", "chest"]
else:
    objs = CONFUSERS; sites = SHEET_SITES

manifest = []
for oname, sp in objs.items():
    path, mat, size, roll = sp["path"], sp["mat"], sp["size"], sp["roll"]
    if not path or not os.path.exists(path or ""):
        print(f"MISSING mesh for {oname}: {path}", flush=True); continue
    print(f"== {oname}  ({os.path.basename(path)}, {mat}, {size*100:.0f}cm, roll {roll}) ==", flush=True)
    threat = load_confuser(path, mat, size)
    for s in sites:
        fn = render(oname, threat, s, roll)
        if fn: manifest.append(dict(object=oname, site=s, material=mat, img=fn))
        print(f"   {s:16s} {'ok' if fn else 'FAIL'}", flush=True)

with open(os.path.join(OUT, "manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=2)
print(f"\nDONE {len(manifest)} renders -> {OUT}", flush=True)
