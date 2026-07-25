"""Dataset 3: 50 bodies x 2 stances x (17 sites + baseline) = 950 scans, at the REAL standoff.

WHAT CHANGED vs dataset2 (shape_driver.py), and why -- all measured 2026-07-16:

  STANDOFF 1.0 -> 2.0 m   dataset2 was moved IN to 1 m to make the gun resolvable
                          (stop_pose_driver.py:73). For a zero-shot study that is backwards:
                          it trains on a domain 2x sharper than the one we test on. The real
                          system's `dv` spans 1.499-2.499 m -> ~2 m. The test domain sets the
                          geometry. Costs resolution (2.2 -> 4.4 cm cross-range); that IS the
                          real difficulty.

  hann -> taylor(sll=40)  hann is Pareto-dominated: taylor is 15% narrower AND 8.6 dB quieter.
                          Not cosmetic -- the gun sits 19-33 dB below the image peak, so hann's
                          -31.5 dB sidelobes land INSIDE the target's dynamic range and a bright
                          body glint can throw a sidelobe as strong as the threat. Measured
                          +1.92 dB median contrast over 34 positives (better in 31/34).

  mag3d -> field3d        COMPLEX, not magnitude. Needed to resample onto the real system's
                          angular grid (201,105,57) coherently. NB backproject's output is
                          carrier-folded (~9.2 cycles per 5 mm bin): fine to STORE (each voxel's
                          xyz is known, so the carrier is removable analytically), FATAL to
                          interpolate without de-ramping first.

  full cube -> +/-5 cm    Range-gated about the per-cell surface depth. 5 cm is set by GRAZING,
       range gate         not by resolution: within a 1 cm cross-range cell an oblique surface
                          spans cell/tan(theta) -- 5.7 cm at 10 deg. A +/-1 cm gate would
                          truncate everything past 45 deg, i.e. wb_left/right, thigh_l/r,
                          ankle_l/r -- 10 of 17 carry sites, and precisely the grazing ones.

  build once per BODY     pose/mesh/sampling depend on neither stance nor standoff; dataset2
                          rebuilt per stance for nothing (~97 s x2 -> x1).

  han_db KEPT IDENTICAL   report.py / report2.py / pair_mesh_recon.py all read it. Dropping
  + han_clutter           han_contrast_db is what broke pair_mesh_recon on the probe output.
  + han_contrast_db

  sig for a SUBSET        Every SIG_EVERY-th scan keeps the raw (256,16900) measurement (~35 MB).
                          Without it, re-windowing or re-gridding means re-running the forward.

NOT included, deliberately: roughness (MATERIALS carries per-material sigma but the driver takes
only [0]) and confusers (threats.build_library already returns phone/keys/belt_buckle/wallet).
Both are known gaps -- see DESIGN.md sec.5.

  DS3_OUT=/path ~/miniconda3/envs/ra/bin/python dataset3_driver.py
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

# Date is FIXED, not datetime.now(): the driver resumes by checking which files exist, so a
# clock-derived path would silently start a fresh dataset if a run crossed midnight.
DS3_NAME = "dataset3_2026-07-16"
OUT = os.environ.get("DS3_OUT", os.path.expanduser(f"~/{DS3_NAME}"))
os.makedirs(OUT, exist_ok=True)
STL = next(p for p in (f"{HERE}/glock17_ext.stl",
                       os.path.expanduser("~/walkthrough/glock17_ext.stl"),
                       "/Users/jonah/Documents/scratch_results/threat_dataset/glock17_ext.stl")
           if os.path.exists(p))
PHONE = os.path.join(os.path.dirname(STL), "phone_hull.stl")

LAM = C0 / 290e9
NFREQ = 256
STANDOFF = float(os.environ.get("DS3_STANDOFF", "2.0"))   # sensor->body. NOT gap_m (clothing).
GATE_T = float(os.environ.get("DS3_GATE", "0.05"))        # +/- m about the surface depth
SIG_EVERY = int(os.environ.get("DS3_SIG_EVERY", "32"))    # keep raw sig every Nth scan
ROLL, GAP = 0.0, 0.012                       # GAP = CLOTHING standoff, NOT sensor standoff
FLIP = {"ankle_l", "thigh_r"}
SITES = {
    "front": ["wb_front_mid", "wb_front_left", "wb_front_right", "belly", "thigh_l", "thigh_r",
              "wb_left", "wb_right", "ankle_l", "ankle_r"],
    "back":  ["wb_back_mid", "wb_back_left", "wb_back_right", "wb_left", "wb_right",
              "ankle_l", "ankle_r"],
}
XS = np.arange(-0.90, 0.9001, 0.01)          # 1 cm: magnitude-Nyquist @2 m is 1.54 cm
ZS = np.arange(-1.05, 1.0501, 0.01)
DY = 0.005                                   # 5 mm = c/2B exactly; matches the real dv step
KGATE = int(round(2 * GATE_T / DY)) + 1      # 21 samples at +/-5 cm

g = np.linspace(-0.067 / 2, 0.067 / 2, 130)
xx, zz = np.meshgrid(g, g, indexing="ij")
RX = np.column_stack([xx.ravel(), np.zeros(130 * 130), zz.ravel()])
TX = np.array([[0.15, 0.0, 0.0]])
FREQ = np.linspace(275e9, 305e9, NFREQ)
KV = 2 * np.pi * FREQ / C0
WINDOW = os.environ.get("DS3_WINDOW", "taylor")
if WINDOW == "taylor":
    from scipy.signal.windows import taylor as _taylor
    APER = _taylor(130, nbar=5, sll=40, norm=False)       # matches recon.py's _DEFAULT_WINDOW
elif WINDOW == "hann":
    APER = np.hanning(130)                                # dataset2 provenance
else:
    sys.exit(f"DS3_WINDOW must be 'taylor' or 'hann', got {WINDOW!r}")
POSE = screening_pose()
PISTOL = threats.build_library(glock_stl=STL, phone_stl=PHONE)["pistol"]
S_SKIN = threats.MATERIALS["skin"][0]        # [0]=s only; [1]=sigma is DISCARDED (no roughness)
SHAPES = json.load(open(f"{HERE}/shapes_5x5.json"))


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
    """Full 3-D COMPLEX cube. Aperture taper via APER; range taper via recon's freq_window."""
    W = np.outer(APER, APER).ravel().astype(np.float32)
    img = backproject(sig * W[None, :], tx=TX, rx=RX, f=FREQ, grid=grid, npad=4096,
                      freq_window=WINDOW, precision="fp32")
    try:
        img = rf.cp.asnumpy(img)
    except Exception:
        pass
    return img                                # complex64 (nx, ny, nz)


def depth_map_of(q_vis):
    """Nearest visible surface y per (x,z) cell = the surface at image resolution. NaN = empty."""
    ix = np.clip(np.round((q_vis[:, 0] - XS[0]) / 0.01).astype(np.int32), 0, len(XS) - 1)
    iz = np.clip(np.round((q_vis[:, 2] - ZS[0]) / 0.01).astype(np.int32), 0, len(ZS) - 1)
    dm = np.full((len(XS), len(ZS)), np.inf)
    np.minimum.at(dm, (ix, iz), q_vis[:, 1])
    dm[np.isinf(dm)] = np.nan
    return dm.astype(np.float32)


def gate(field, grid, dmap):
    """Range-gate the cube about the per-cell surface depth -> (n_cells, KGATE) complex64.

    Indices are IMPLICIT: recoverable from dmap + ys[0] + DY, so no index array is stored.
    Grid-aligned (the gate is a subset of the cube's own voxels), so this is a LOSSLESS subset --
    backproject_points at these coords returns the identical values (rel-err < 1e-6, see
    recon.test_backproject_points_matches_grid).
    """
    cells = np.flatnonzero(np.isfinite(dmap.ravel()))
    ix, iz = np.unravel_index(cells, dmap.shape)
    iy0 = np.round((dmap.ravel()[cells] - grid.ys[0]) / DY).astype(np.int32)
    off = np.arange(KGATE, dtype=np.int32) - KGATE // 2
    iy = np.clip(iy0[:, None] + off[None, :], 0, len(grid.ys) - 1)     # (n_cells, KGATE)
    return field[ix[:, None], iy, iz[:, None]].astype(np.complex64), cells.astype(np.int32)


print(f"[{datetime.datetime.now():%H:%M}] DATASET3: {len(SHAPES)} bodies x 2 stances "
      f"x (17 sites + baseline) = {len(SHAPES)*19} scans", flush=True)
print(f"  standoff={STANDOFF} m  window={WINDOW}  gate=+/-{GATE_T*100:.0f} cm (K={KGATE})  "
      f"sig every {SIG_EVERY}  GPU={rf._HAS_CUPY} -> {OUT}", flush=True)

nscan = 0
t_start = time.time()
for bi, shp in enumerate(SHAPES):
    bid = bid_of(shp)
    if all(os.path.exists(f"{OUT}/ds3_{bid}_{st}_{s}.npz")
           for st in ("front", "back") for s in SITES[st] + ["nothreat"]):
        print(f"  skip {bid} (done)", flush=True); nscan += 19; continue

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
        sig_body = fwd(bq[keep_b0], sb[keep_b0])

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

        for site in ["nothreat"] + SITES[stance]:
            t1 = time.time()
            f_out = f"{OUT}/ds3_{bid}_{stance}_{site}.npz"
            if os.path.exists(f_out):
                nscan += 1; continue

            if site == "nothreat":
                sig, ng, n_seated, pl = sig_body, 0, 0, None
                gq = kt = None
            else:
                pl = anatomy.sample_placement(model, site, np.random.default_rng(0), jitter=False)
                pl.update(roll=ROLL, standoff=GAP, flip=site in FLIP)
                gq, gn, gs = attach.attach(PISTOL, Vp, VC, F, pl, target_nn=0.125 * LAM,
                                           vert_normals=m.vertex_normals)
                gq = to_sim(gq, turn, zc); gn = to_sim(gn, turn, zc, is_dir=True)
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

            field = recon_complex(sig, grid)            # COMPLEX cube
            mag = np.abs(field)
            pk = float(mag.max())
            db = (20*np.log10(np.maximum(mag.max(1), pk*1e-4)/pk)).astype(np.float32)  # han_db
            fg, cells = gate(field, grid, dmap)

            rec = dict(site=site, han_db=db, han_peak=pk,
                       field_gated=fg, gate_cells=cells, n_seated=n_seated, n_visible=ng,
                       **common)
            if ng:
                gk = gq[kt]
                gx = np.clip(np.round((gk[:,0]-XS[0])/0.01).astype(np.int32), 0, len(XS)-1)
                gy = np.clip(np.round((gk[:,1]-grid.ys[0])/DY).astype(np.int32), 0, len(grid.ys)-1)
                gz = np.clip(np.round((gk[:,2]-ZS[0])/0.01).astype(np.int32), 0, len(ZS)-1)
                gun_lin = float(np.percentile(mag[gx, gy, gz], 90))
                gc = gk.mean(0)
                X, Y, Z = np.meshgrid(grid.xs, grid.ys, grid.zs, indexing="ij")
                r = np.sqrt((X-gc[0])**2 + (Y-gc[1])**2 + (Z-gc[2])**2)
                ann = (r > 0.08) & (r < 0.25)
                clut = float(np.percentile(mag[ann], 90)) if ann.any() else np.nan
                del X, Y, Z, r, ann
                rec.update(face=pl["face"], bary=np.array(pl["bary"]), flip=bool(pl["flip"]),
                           seat_xyz=gc, gun_aabb=np.vstack([gk.min(0), gk.max(0)]),
                           gun_vox=np.unique(np.column_stack([gx, gy, gz]), axis=0).astype(np.int16),
                           visibility=ng/n_seated,
                           obliquity_deg=float(np.degrees(np.arccos(
                               np.clip(np.abs(-gn[kt][:,1]).mean(), 0, 1)))),
                           han_gun_db=20*np.log10(max(gun_lin, pk*1e-5)/pk),
                           han_clutter=clut, han_contrast_db=20*np.log10(gun_lin/(clut+1e-30)))
            else:
                rec.update(seat_xyz=np.full(3, np.nan), gun_vox=np.zeros((0,3), np.int16),
                           gun_aabb=np.full((2,3), np.nan), visibility=np.nan,
                           obliquity_deg=np.nan, han_gun_db=np.nan, han_clutter=np.nan,
                           han_contrast_db=np.nan)
            if nscan % SIG_EVERY == 0:
                rec["sig"] = sig.astype(np.complex64)   # the render-anything reference
            np.savez_compressed(f_out, **rec)
            nscan += 1
            c = rec.get("han_contrast_db", np.nan)
            print(f"    {bid}/{stance}/{site:<14} vis {rec['visibility'] if ng else 0:.3f} "
                  f"contrast {c:+6.1f} dB  gated {fg.shape}  ({time.time()-t1:.0f}s)"
                  + ("  +sig" if "sig" in rec else ""), flush=True)
        del bq, sb, sig_body
    del bq_raw, Vp, F, m, Vs, Fs

print(f"[{datetime.datetime.now():%H:%M}] DATASET3 done: {nscan} scans in "
      f"{(time.time()-t_start)/60:.0f} min -> {OUT}", flush=True)
