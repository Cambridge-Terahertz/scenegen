"""Dataset 4: threat vs REAL benign objects, 98 bodies, at the dataset3 geometry/settings.

WHAT CHANGED vs dataset3 (dataset3_driver.py), and why:

  ONE THREAT, FOUR       dataset3 was gun-vs-BARE-BODY: 850 positives against 100 empty-body
  BENIGN OBJECTS         negatives. A detector trained on that learns "is there an object on
                         this body", not "is that object a gun" -- and the real system's false
                         alarms come from phones, wallets, buckles and badges, not bare skin.
                         dataset4's negatives are mostly BODIES CARRYING SOMETHING BENIGN, so
                         the decision boundary is forced onto object identity.
                         Threat  = the SAME Glock 17 used in dataset3 (glock17_ext.stl).
                         Benign  = phone / wallet / buckle / name_badge, built from real CAD in
                         ~/Documents/Blender/Models by objects_v4 (see objects_v4/manifest.json).

  98 BODIES (7x7)        dataset3's 50 (5x5) gave only 100 negatives, and the plan repeatedly
                         flags that as too thin to falsify a detector (dataset2's 36 negatives
                         produced a shuffle-control of 0.610 vs a 0.514 signal). 2 x 7 x 7 = 98
                         bodies, same sample->measure->stratify pool (seed 7, 600/gender).

  BALANCED LABELS        1,666 threat / 2,254 no-threat = 3,920 scans, vs dataset3's 850/100.

  phase_recursion ON     raforward 487d417 (2026-07-21). MEASURED ON THIS 5090, not assumed:
                         fast-math alone is only 1.06x here (the commit's 1.49x was an A4000),
                         but phase_recursion is 1.86x, for 2.34e-3 field relerr -- the commit
                         measured recon correlation 0.9999977 -> 0.9999956, peak voxel unchanged.
                         Set RA_PHASE_RECURSION=0 to disable.

UNCHANGED from dataset3 (deliberately -- this set must be comparable to it): 2.0 m standoff,
taylor(sll=40) on both tapers, +/-5 cm complex range gate, 1 cm cross-range grid, 5 mm range
step, the body-blocked delta forward, and the han_db definition.

STILL NOT INCLUDED (the known fidelity gap, unchanged from dataset3): roughness and a noise
floor. MATERIALS carries a per-material sigma and the driver still takes only [0]. Until
plan sec.7-step-2's acceptance tests pass, this remains a SPECULAR dataset -- good for object
discrimination, not yet a trustworthy sim->real headline. See DESIGN.md sec.5.

FIELD NAMING: the seated object is recorded as obj_* (obj_name/obj_vox/obj_aabb/...). For
backward compatibility with the dataset3 analysis helpers (ds3_voxels.py etc.), threat scans
ALSO write the gun_* aliases; benign scans leave them empty.

  DS4_OUT=/path ~/miniconda3/envs/ra/bin/python dataset4_driver.py
"""
import datetime
import json
import os
import sys
import time

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir("/Users/jonah"):
    os.environ.setdefault("SMPLX_MODELS", os.path.expanduser("~/smplx_models"))
    os.environ.setdefault("AMASS_ROOT", os.path.expanduser("~/amass"))
# Enable the 2026-07-21 forward-kernel phase recursion unless explicitly turned off.
os.environ.setdefault("RA_PHASE_RECURSION", "1")
for p in (os.path.expanduser("~/walkthrough"),
          os.path.expanduser("~/Documents/code/model_gen/python"),
          "/Users/jonah/Documents/scratch_results/threat_dataset"):
    if os.path.isdir(p):
        sys.path.insert(0, p)
sys.path.insert(0, HERE)                     # our own copies MUST win

from mesh_sample import sample_mesh                      # noqa: E402
from bodies_smplx import subdivide_front                 # noqa: E402
import anatomy, attach, threats, body as bodylib         # noqa: E402
import raforward as rf                                   # noqa: E402
from raforward import Scene, C0                          # noqa: E402
from recon import ReconGrid, backproject                 # noqa: E402
from bodies import zbuffer_cull                          # noqa: E402
from stance import screening_pose                        # noqa: E402

if not rf._HAS_CUPY:
    sys.exit("FATAL: no CuPy -- use ~/miniconda3/envs/ra/bin/python, not system python3.")

DS4_NAME = "dataset4_2026-07-21"
OUT = os.environ.get("DS4_OUT", os.path.expanduser(f"~/{DS4_NAME}"))
os.makedirs(OUT, exist_ok=True)
STL = next(p for p in (f"{HERE}/glock17_ext.stl",
                       os.path.expanduser("~/walkthrough/glock17_ext.stl"),
                       "/Users/jonah/Documents/scratch_results/threat_dataset/glock17_ext.stl")
           if os.path.exists(p))
PHONE = os.path.join(os.path.dirname(STL), "phone_hull.stl")
OBJDIR = next(p for p in (f"{HERE}/objects_v4",
                          "/Users/jonah/Documents/scratch_results/threat_dataset/objects_v4")
              if os.path.isdir(p))

LAM = C0 / 290e9
NFREQ = 256
STANDOFF = float(os.environ.get("DS4_STANDOFF", "2.0"))
GATE_T = float(os.environ.get("DS4_GATE", "0.05"))
SIG_EVERY = int(os.environ.get("DS4_SIG_EVERY", "128"))   # ~31 raw-sig scans, ~1.1 GB
ROLL, GAP = 0.0, 0.012
FLIP = {"ankle_l", "thigh_r"}                # pistol orientation only (see note in loop)
XS = np.arange(-0.90, 0.9001, 0.01)
ZS = np.arange(-1.05, 1.0501, 0.01)
DY = 0.005
KGATE = int(round(2 * GATE_T / DY)) + 1

g = np.linspace(-0.067 / 2, 0.067 / 2, 130)
xx, zz = np.meshgrid(g, g, indexing="ij")
RX = np.column_stack([xx.ravel(), np.zeros(130 * 130), zz.ravel()])
TX = np.array([[0.15, 0.0, 0.0]])
FREQ = np.linspace(275e9, 305e9, NFREQ)
KV = 2 * np.pi * FREQ / C0
WINDOW = os.environ.get("DS4_WINDOW", "taylor")
if WINDOW == "taylor":
    from scipy.signal.windows import taylor as _taylor
    APER = _taylor(130, nbar=5, sll=40, norm=False)
elif WINDOW == "hann":
    APER = np.hanning(130)
else:
    sys.exit(f"DS4_WINDOW must be 'taylor' or 'hann', got {WINDOW!r}")
POSE = screening_pose()
S_SKIN = threats.MATERIALS["skin"][0]        # [0]=s only; [1]=sigma DISCARDED (no roughness)
SHAPES = json.load(open(f"{HERE}/shapes_7x7.json"))
_NB = int(os.environ.get("DS4_NBODIES", "0"))    # >0 truncates the body list (smoke tests)
if _NB:
    SHAPES = SHAPES[:_NB]

# ---------------------------------------------------------------- the object catalogue -----
# The benign objects reuse EXISTING MATERIALS entries whose s already matches the value the
# mesh-prep step assigned, so threats.py needs no edit (another session is editing it today):
#   phone -> alloy 0.95 | wallet -> plastic 0.30 | buckle -> steel 1.00 | badge -> polymer 0.35
#
# ALL FOUR ARE RIGID -- do not set conformal=True here. attach.attach_conformal returns the
# BODY vertex normal for every object point (obj_nrm is accepted and then never used), so
# attach.front_cull's `n @ view_dir > 0` test passes for EVERY point and the slab's occluded
# BACK face survives and radiates. Measured in the 1-body smoke test: conformal wallet and
# name_badge reported visibility EXACTLY 1.000 at all 7 placements, while the rigid phone /
# buckle / pistol gave the ~0.5 a closed surface must give. That is the same double-counting
# front_cull was written to prevent (see its docstring). It inflates contrast to +38..+52 dB
# AND injects a phantom surface ~1.1 cm behind the real one -- inside the +/-5 cm gate, so it
# corrupts RANGE structure, which is the one feature family that worked on dataset3.
BENIGN_SPEC = {                       # name: (stl, material, conformal)
    "phone":      ("phone.stl",      "alloy",   False),
    "wallet":     ("wallet.stl",     "plastic", False),
    "buckle":     ("buckle.stl",     "steel",   False),
    "name_badge": ("name_badge.stl", "polymer", False),
}
OBJECTS = {"pistol": (threats.build_library(glock_stl=STL, phone_stl=PHONE)["pistol"], True)}
for _n, (_f, _mat, _conf) in BENIGN_SPEC.items():
    OBJECTS[_n] = (threats.Threat(_n, "benign", _mat,
                                  mesh=trimesh.load(f"{OBJDIR}/{_f}", process=False),
                                  conformal=_conf), False)

# Where each object is carried. The pistol keeps dataset3's 17 sites verbatim so the threat
# class is directly comparable; benign objects go where people actually carry them.
SITES = {
    "pistol": {"front": ["wb_front_mid", "wb_front_left", "wb_front_right", "belly",
                         "thigh_l", "thigh_r", "wb_left", "wb_right", "ankle_l", "ankle_r"],
               "back":  ["wb_back_mid", "wb_back_left", "wb_back_right", "wb_left",
                         "wb_right", "ankle_l", "ankle_r"]},
    "phone":      {"front": ["thigh_l", "thigh_r", "wb_front_left"],
                   "back":  ["wb_back_left", "wb_back_right"]},
    "wallet":     {"front": ["thigh_l", "thigh_r"],
                   "back":  ["wb_back_left", "wb_back_right", "wb_back_mid"]},
    "buckle":     {"front": ["wb_front_mid", "belly", "wb_front_left", "wb_front_right"],
                   "back":  ["wb_back_mid"]},
    "name_badge": {"front": ["chest", "wb_front_mid", "wb_front_left", "wb_front_right"],
                   "back":  ["upper_back", "wb_back_mid"]},
}


def placements(stance):
    """[(obj_name, site)] for a stance, baseline first so it lands even if a run is cut short."""
    out = [("none", "nothreat")]
    for name in OBJECTS:
        out += [(name, s) for s in SITES[name][stance]]
    return out


def bid_of(shp):
    return f"{shp['gender'][0]}_{shp['height']}_{shp['build']}"


def to_sim(P, turn, z_center, is_dir=False):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    if turn:
        x, z = -x, -z
    if is_dir:
        return np.column_stack([x, -z, y])
    return np.column_stack([x, STANDOFF - z, y - z_center])


def fwd(q, s):
    sig = np.zeros((NFREQ, len(RX)), np.complex64)
    for c0 in range(0, len(q), 2_000_000):
        sig += rf.forward(Scene(q=q[c0:c0 + 2_000_000], s=s[c0:c0 + 2_000_000],
                                tx=TX, rx=RX, k=KV), precision="fp32")
    return sig


def mask_of(idx, n):
    m = np.zeros(n, bool)
    m[np.asarray(idx).ravel()] = True
    return m


def recon_complex(sig, grid):
    W = np.outer(APER, APER).ravel().astype(np.float32)
    img = backproject(sig * W[None, :], tx=TX, rx=RX, f=FREQ, grid=grid, npad=4096,
                      freq_window=WINDOW, precision="fp32")
    try:
        img = rf.cp.asnumpy(img)
    except Exception:
        pass
    return img


def depth_map_of(q_vis):
    ix = np.clip(np.round((q_vis[:, 0] - XS[0]) / 0.01).astype(np.int32), 0, len(XS) - 1)
    iz = np.clip(np.round((q_vis[:, 2] - ZS[0]) / 0.01).astype(np.int32), 0, len(ZS) - 1)
    dm = np.full((len(XS), len(ZS)), np.inf)
    np.minimum.at(dm, (ix, iz), q_vis[:, 1])
    dm[np.isinf(dm)] = np.nan
    return dm.astype(np.float32)


def gate(field, grid, dmap):
    """Range-gate the cube about the per-cell surface depth -> (n_cells, KGATE) complex64.
    Indices are IMPLICIT (recoverable from dmap + ys[0] + DY). Grid-aligned = lossless subset."""
    cells = np.flatnonzero(np.isfinite(dmap.ravel()))
    ix, iz = np.unravel_index(cells, dmap.shape)
    iy0 = np.round((dmap.ravel()[cells] - grid.ys[0]) / DY).astype(np.int32)
    off = np.arange(KGATE, dtype=np.int32) - KGATE // 2
    iy = np.clip(iy0[:, None] + off[None, :], 0, len(grid.ys) - 1)
    return field[ix[:, None], iy, iz[:, None]].astype(np.complex64), cells.astype(np.int32)


N_PER_BODY = sum(len(placements(st)) for st in ("front", "back"))
N_THREAT = sum(len(SITES["pistol"][st]) for st in ("front", "back"))
print(f"[{datetime.datetime.now():%H:%M}] DATASET4: {len(SHAPES)} bodies x {N_PER_BODY} "
      f"placements = {len(SHAPES)*N_PER_BODY} scans "
      f"({len(SHAPES)*N_THREAT} threat / {len(SHAPES)*(N_PER_BODY-N_THREAT)} no-threat)",
      flush=True)
print(f"  objects: " + ", ".join(f"{n}{'*' if t else ''}" for n, (_, t) in OBJECTS.items())
      + "   (* = threat)", flush=True)
print(f"  standoff={STANDOFF} m  window={WINDOW}  gate=+/-{GATE_T*100:.0f} cm (K={KGATE})  "
      f"sig every {SIG_EVERY}  phase_recursion={os.environ['RA_PHASE_RECURSION']}  "
      f"GPU={rf._HAS_CUPY} -> {OUT}", flush=True)

nscan = 0
t_start = time.time()
for bi, shp in enumerate(SHAPES):
    bid = bid_of(shp)
    if all(os.path.exists(f"{OUT}/ds4_{bid}_{st}_{o}_{s}.npz")
           for st in ("front", "back") for o, s in placements(st)):
        print(f"  skip {bid} (done)", flush=True)
        nscan += N_PER_BODY
        continue

    t0 = time.time()
    betas = np.array(shp["betas"])
    model = bodylib.get_model(shp["gender"])
    VC = bodylib.canonical_verts(model)
    # ---- BUILD ONCE PER BODY: independent of stance AND standoff ------------------------
    Vp, F = bodylib.posed_verts(model, betas=betas, body_pose=POSE)
    m = trimesh.Trimesh(vertices=Vp, faces=F, process=False)
    m.fix_normals()
    zc = 0.5 * (Vp[:, 1].min() + Vp[:, 1].max())
    Vs, Fs, _ = subdivide_front(m, 0.5 * LAM)
    bq_raw, _, _ = sample_mesh(Vs, Fs, 0.125 * LAM, front_cull=False)
    el = (time.time() - t_start) / 60
    print(f"[{datetime.datetime.now():%H:%M}] BODY {bi+1}/{len(SHAPES)} {bid}: "
          f"{len(bq_raw):,} pts ({time.time()-t0:.0f}s)  [{el:.0f} min elapsed]", flush=True)

    for stance in ("front", "back"):
        t0 = time.time()
        turn = stance == "back"
        bq = to_sim(bq_raw.astype(np.float64), turn, zc)
        sb = np.full(len(bq), S_SKIN)
        keep_b0 = mask_of(zbuffer_cull(bq, pixel=0.003, keep_depth=0.02), len(bq))
        sig_body = fwd(bq[keep_b0], sb[keep_b0])          # THE expensive step, reused below

        ylo, yhi = bq[keep_b0][:, 1].min(), bq[keep_b0][:, 1].max()
        grid = ReconGrid(xs=XS, ys=np.arange(ylo - 0.05, yhi + 0.05, DY), zs=ZS)
        dmap = depth_map_of(bq[keep_b0])
        common = dict(body=bid, stance=stance, standoff_m=STANDOFF, window=WINDOW,
                      gate_t=GATE_T, kgate=KGATE, **shp,
                      xs=XS.astype(np.float32), ys=grid.ys.astype(np.float32),
                      zs=ZS.astype(np.float32), depth_map=dmap, gap_m=GAP,
                      ext=np.array([XS[0]*100, XS[-1]*100, ZS[0]*100, ZS[-1]*100]))
        print(f"  {bid}/{stance}: {int(keep_b0.sum()):,} visible, grid {grid.shape}, "
              f"{time.time()-t0:.0f}s", flush=True)

        for obj_name, site in placements(stance):
            t1 = time.time()
            f_out = f"{OUT}/ds4_{bid}_{stance}_{obj_name}_{site}.npz"
            if os.path.exists(f_out):
                nscan += 1
                continue

            if obj_name == "none":
                sig, ng, n_seated, pl, is_threat = sig_body, 0, 0, None, False
                gq = kt = gn = None
            else:
                obj, is_threat = OBJECTS[obj_name]
                pl = anatomy.sample_placement(model, site, np.random.default_rng(0),
                                              jitter=False)
                # FLIP is a pistol-orientation correction (which side the grip faces); it is
                # meaningless for the near-symmetric benign slabs, so only the gun uses it.
                pl.update(roll=ROLL, standoff=GAP,
                          flip=(is_threat and site in FLIP))
                gq, gn, gs = attach.attach(obj, Vp, VC, F, pl, target_nn=0.125 * LAM,
                                           vert_normals=m.vertex_normals)
                gq = to_sim(gq, turn, zc)
                gn = to_sim(gn, turn, zc, is_dir=True)
                n_seated = len(gq)
                gq, gn, gs = attach.front_cull(gq, gn, gs, view_dir=(0.0, -1.0, 0.0))
                q_all = np.vstack([bq, gq])
                keep = mask_of(zbuffer_cull(q_all, pixel=0.003, keep_depth=0.02), len(q_all))
                kb, kt = keep[:len(bq)], keep[len(bq):]
                ng = int(kt.sum())
                shadow = keep_b0 & ~kb                  # DESIGN.md sec.2: load-bearing
                sig = sig_body.copy()
                if shadow.any():
                    sig -= fwd(bq[shadow], sb[shadow])
                if ng:
                    sig += fwd(gq[kt], gs[kt])

            field = recon_complex(sig, grid)
            mag = np.abs(field)
            pk = float(mag.max())
            db = (20*np.log10(np.maximum(mag.max(1), pk*1e-4)/pk)).astype(np.float32)
            fg, cells = gate(field, grid, dmap)

            rec = dict(obj_name=obj_name, is_threat=bool(is_threat), site=site,
                       han_db=db, han_peak=pk, field_gated=fg, gate_cells=cells,
                       n_seated=n_seated, n_visible=ng, **common)
            if ng:
                gk = gq[kt]
                gx = np.clip(np.round((gk[:,0]-XS[0])/0.01).astype(np.int32), 0, len(XS)-1)
                gy = np.clip(np.round((gk[:,1]-grid.ys[0])/DY).astype(np.int32), 0,
                             len(grid.ys)-1)
                gz = np.clip(np.round((gk[:,2]-ZS[0])/0.01).astype(np.int32), 0, len(ZS)-1)
                obj_lin = float(np.percentile(mag[gx, gy, gz], 90))
                gc = gk.mean(0)
                X, Y, Z = np.meshgrid(grid.xs, grid.ys, grid.zs, indexing="ij")
                r = np.sqrt((X-gc[0])**2 + (Y-gc[1])**2 + (Z-gc[2])**2)
                ann = (r > 0.08) & (r < 0.25)
                clut = float(np.percentile(mag[ann], 90)) if ann.any() else np.nan
                del X, Y, Z, r, ann
                obj_vox = np.unique(np.column_stack([gx, gy, gz]), axis=0).astype(np.int16)
                obj_aabb = np.vstack([gk.min(0), gk.max(0)])
                rec.update(face=pl["face"], bary=np.array(pl["bary"]), flip=bool(pl["flip"]),
                           seat_xyz=gc, obj_aabb=obj_aabb, obj_vox=obj_vox,
                           visibility=ng/n_seated,
                           obliquity_deg=float(np.degrees(np.arccos(
                               np.clip(np.abs(-gn[kt][:,1]).mean(), 0, 1)))),
                           han_obj_db=20*np.log10(max(obj_lin, pk*1e-5)/pk),
                           han_clutter=clut, han_contrast_db=20*np.log10(obj_lin/(clut+1e-30)))
                # dataset3-compatible aliases, threat scans only (ds3_voxels.py reads these)
                rec.update(gun_vox=obj_vox if is_threat else np.zeros((0, 3), np.int16),
                           gun_aabb=obj_aabb if is_threat else np.full((2, 3), np.nan),
                           han_gun_db=rec["han_obj_db"] if is_threat else np.nan)
            else:
                rec.update(seat_xyz=np.full(3, np.nan), obj_vox=np.zeros((0,3), np.int16),
                           obj_aabb=np.full((2,3), np.nan), visibility=np.nan,
                           obliquity_deg=np.nan, han_obj_db=np.nan, han_clutter=np.nan,
                           han_contrast_db=np.nan, gun_vox=np.zeros((0,3), np.int16),
                           gun_aabb=np.full((2,3), np.nan), han_gun_db=np.nan)
            if nscan % SIG_EVERY == 0:
                rec["sig"] = sig.astype(np.complex64)
            np.savez_compressed(f_out, **rec)
            nscan += 1
            print(f"    {bid}/{stance}/{obj_name:<10}/{site:<14} "
                  f"{'THREAT' if is_threat else 'benign':<6} vis "
                  f"{rec['visibility'] if ng else 0:.3f} contrast "
                  f"{rec['han_contrast_db']:+6.1f} dB  ({time.time()-t1:.0f}s)"
                  + ("  +sig" if "sig" in rec else ""), flush=True)
        del bq, sb, sig_body
    del bq_raw, Vp, F, m, Vs, Fs

print(f"[{datetime.datetime.now():%H:%M}] DATASET4 done: {nscan} scans in "
      f"{(time.time()-t_start)/60:.0f} min -> {OUT}", flush=True)
