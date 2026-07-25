"""PROBE: does 3D separate the gun from body glint where the 2D max-projection cannot?

Measured on dataset2 (2D han_db only, STANDOFF=1.0) 2026-07-16:
  * every brightness-scalar feature is at CHANCE (0.45-0.54); the gun is 19-33 dB
    below the image peak and is the brightest thing in only 12/306 scans.
  * BUT per-blob, gun vs body glint is 0.968-separable by pure SHAPE (area 119 vs
    22 px, aspect 1.89 vs 1.17) -- once you know where the gun is.
  * per-image, without that oracle: 0.514, and the shuffled-label control scored
    0.610, so the current dataset can neither validate nor refute a detector.

Hypothesis: the gun sits GAP=1.2 cm proud of the skin ~= 1.7 range cells at the
~7 mm Hann range resolution, and `han_db = mag.max(1)` collapses exactly that
axis. So depth may separate gun from glint where the projection cannot. UNTESTED
-- dataset2 stores no 3D.

This probe stops throwing the cube away. `recon()` already computes the full 3D
magnitude cube and discards it one line later; we save it, plus the range axis,
the gun's voxel footprint and a depth map. It also runs BOTH standoffs so we can
see whether 2 m loses glints (specular acceptance ~halves) while closing the
resolution gap.

  1 body x 2 stances x 2 standoffs x (17 sites + baseline) ~= 38 scans, ~10 min.

  PROBE_OUT=/path PROBE_BODY=m_mid_mid ~/miniconda3/envs/ra/bin/python probe_driver.py
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
sys.path.insert(0, HERE)                     # our own copies MUST win -- see driver notes

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

OUT = os.environ.get("PROBE_OUT", os.path.expanduser("~/probe_out"))
os.makedirs(OUT, exist_ok=True)
STL = next(p for p in (f"{HERE}/glock17_ext.stl",
                       os.path.expanduser("~/walkthrough/glock17_ext.stl"),
                       "/Users/jonah/Documents/scratch_results/threat_dataset/glock17_ext.stl")
           if os.path.exists(p))
PHONE = os.path.join(os.path.dirname(STL), "phone_hull.stl")

LAM = C0 / 290e9
NFREQ = 256
STANDOFFS = (1.0, 2.0)                       # <-- the comparison. 1.0 = as-built, 2.0 = real system
ROLL, GAP = 0.0, 0.012                       # GAP = CLOTHING standoff, NOT sensor standoff
FLIP = {"ankle_l", "thigh_r"}
SITES = {
    "front": ["wb_front_mid", "wb_front_left", "wb_front_right", "belly", "thigh_l", "thigh_r",
              "wb_left", "wb_right", "ankle_l", "ankle_r"],
    "back":  ["wb_back_mid", "wb_back_left", "wb_back_right", "wb_left", "wb_right",
              "ankle_l", "ankle_r"],
}
XS = np.arange(-0.90, 0.9001, 0.01)
ZS = np.arange(-1.05, 1.0501, 0.01)

g = np.linspace(-0.067 / 2, 0.067 / 2, 130)
xx, zz = np.meshgrid(g, g, indexing="ij")
RX = np.column_stack([xx.ravel(), np.zeros(130 * 130), zz.ravel()])
TX = np.array([[0.15, 0.0, 0.0]])
FREQ = np.linspace(275e9, 305e9, NFREQ)
KV = 2 * np.pi * FREQ / C0
# Aperture taper. Hann (as shipped in shape_driver) is Pareto-DOMINATED -- measured on
# this exact N=130 aperture (3 dB width in bins / highest sidelobe past the first null):
#     rectangular 0.906 / -13.3 | hann 1.469 / -31.5 | taylor40 1.250 / -40.1
# taylor(sll=40) is 15% NARROWER *and* 8.6 dB quieter. sll=40 is not arbitrary: the gun
# sits 19-33 dB below the image peak (a body glint), so hann's -31.5 dB sidelobes land
# INSIDE the target's own dynamic range -- a glint can throw a sidelobe as bright as the
# threat. Matches recon.py's default. PROBE_WINDOW=hann reproduces dataset2 bit-for-bit.
WINDOW = os.environ.get("PROBE_WINDOW", "taylor")
if WINDOW == "taylor":
    from scipy.signal.windows import taylor as _taylor
    APER = _taylor(130, nbar=5, sll=40, norm=False)
elif WINDOW == "hann":
    APER = np.hanning(130)
else:
    sys.exit(f"PROBE_WINDOW must be 'taylor' or 'hann', got {WINDOW!r}")
POSE = screening_pose()
PISTOL = threats.build_library(glock_stl=STL, phone_stl=PHONE)["pistol"]
S_SKIN = threats.MATERIALS["skin"][0]
SHAPES = json.load(open(f"{HERE}/shapes.json"))

WANT = os.environ.get("PROBE_BODY", "")
def bid_of(shp):
    return f"{shp['gender'][0]}_{shp['height']}_{shp['build']}"
SHAPES = [s for s in SHAPES if bid_of(s) == WANT] or SHAPES[len(SHAPES) // 2:len(SHAPES) // 2 + 1]


def to_sim(P, turn, z_center, standoff, is_dir=False):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    if turn:
        x, z = -x, -z
    if is_dir:
        return np.column_stack([x, -z, y])
    return np.column_stack([x, standoff - z, y - z_center])


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


def recon(sig, grid):
    """Full 3-D magnitude cube. Same path as shape_driver.recon, but WINDOW-tapered
    on both axes: the aperture via APER, the range via recon's freq_window."""
    W = np.outer(APER, APER).ravel().astype(np.float32)
    img = backproject(sig * W[None, :], tx=TX, rx=RX, f=FREQ, grid=grid, npad=4096,
                      freq_window=WINDOW, precision="fp32")
    try:
        img = rf.cp.asnumpy(img)
    except Exception:
        pass
    return np.abs(img)


def vox_idx(q, grid):
    """(x,z) at 1 cm and y at 5 mm -> integer voxel indices, clipped to the grid."""
    ix = np.clip(np.round((q[:, 0] - XS[0]) / 0.01).astype(np.int32), 0, len(XS) - 1)
    iy = np.clip(np.round((q[:, 1] - grid.ys[0]) / 0.005).astype(np.int32), 0, len(grid.ys) - 1)
    iz = np.clip(np.round((q[:, 2] - ZS[0]) / 0.01).astype(np.int32), 0, len(ZS) - 1)
    return ix, iy, iz


def depth_map_of(q_vis):
    """Nearest visible surface y per (x,z) cell -- the surface at image resolution."""
    ix = np.clip(np.round((q_vis[:, 0] - XS[0]) / 0.01).astype(np.int32), 0, len(XS) - 1)
    iz = np.clip(np.round((q_vis[:, 2] - ZS[0]) / 0.01).astype(np.int32), 0, len(ZS) - 1)
    dm = np.full((len(XS), len(ZS)), np.inf, np.float64)
    np.minimum.at(dm, (ix, iz), q_vis[:, 1])
    dm[np.isinf(dm)] = np.nan
    return dm.astype(np.float32)


print(f"[{datetime.datetime.now():%H:%M}] PROBE: bodies={[bid_of(s) for s in SHAPES]} "
      f"standoffs={STANDOFFS} window={WINDOW} GPU={rf._HAS_CUPY} -> {OUT}", flush=True)

for shp in SHAPES:
    bid = bid_of(shp)
    betas = np.array(shp["betas"])
    model = bodylib.get_model(shp["gender"])
    VC = bodylib.canonical_verts(model)

    for stance in ("front", "back"):
        t0 = time.time()
        turn = stance == "back"
        # ---- build ONCE: pose/mesh/sampling do NOT depend on standoff -------------------
        Vp, F = bodylib.posed_verts(model, betas=betas, body_pose=POSE)
        m = trimesh.Trimesh(vertices=Vp, faces=F, process=False)
        m.fix_normals()
        zc = 0.5 * (Vp[:, 1].min() + Vp[:, 1].max())
        Vs, Fs, _ = subdivide_front(m, 0.5 * LAM)
        bq_raw, _, _ = sample_mesh(Vs, Fs, 0.125 * LAM, front_cull=False)
        print(f"[{datetime.datetime.now():%H:%M}] BUILD {bid}/{stance}: {len(bq_raw):,} pts "
              f"({time.time()-t0:.0f}s)", flush=True)

        for standoff in STANDOFFS:
            t0 = time.time()
            tag = f"{int(round(standoff*100))}cm"
            bq = to_sim(bq_raw.astype(np.float64), turn, zc, standoff)
            sb = np.full(len(bq), S_SKIN)

            keep_b0 = mask_of(zbuffer_cull(bq, pixel=0.003, keep_depth=0.02), len(bq))
            sig_body = fwd(bq[keep_b0], sb[keep_b0])
            print(f"[{datetime.datetime.now():%H:%M}] BLOCK {bid}/{stance}/{tag}: "
                  f"{int(keep_b0.sum()):,} visible, {time.time()-t0:.0f}s", flush=True)

            ylo, yhi = bq[keep_b0][:, 1].min(), bq[keep_b0][:, 1].max()
            grid = ReconGrid(xs=XS, ys=np.arange(ylo - 0.05, yhi + 0.05, 0.005), zs=ZS)
            dmap = depth_map_of(bq[keep_b0])
            common = dict(body=bid, stance=stance, standoff_m=float(standoff),
                          window=WINDOW, **shp,
                          xs=XS.astype(np.float32), ys=grid.ys.astype(np.float32),
                          zs=ZS.astype(np.float32), depth_map=dmap, gap_m=GAP,
                          ext=np.array([XS[0] * 100, XS[-1] * 100, ZS[0] * 100, ZS[-1] * 100]))

            # ---- no-threat baseline, free from the block -------------------------------
            mag = recon(sig_body, grid)
            pk = float(mag.max())
            np.savez_compressed(
                f"{OUT}/pb_{bid}_{stance}_{tag}_nothreat.npz", site="none",
                han_db=(20 * np.log10(np.maximum(mag.max(1), pk * 1e-4) / pk)).astype(np.float32),
                mag3d=mag.astype(np.float32), han_peak=pk,
                n_seated=0, n_visible=0, visibility=np.nan, seat_xyz=np.full(3, np.nan),
                gun_vox=np.zeros((0, 3), np.int16), gun_aabb=np.full((2, 3), np.nan), **common)

            for site in SITES[stance]:
                t1 = time.time()
                pl = anatomy.sample_placement(model, site, np.random.default_rng(0), jitter=False)
                pl.update(roll=ROLL, standoff=GAP, flip=site in FLIP)
                gq, gn, gs = attach.attach(PISTOL, Vp, VC, F, pl, target_nn=0.125 * LAM,
                                           vert_normals=m.vertex_normals)
                gq = to_sim(gq, turn, zc, standoff)
                gn = to_sim(gn, turn, zc, standoff, is_dir=True)
                n_seated = len(gq)
                gq, gn, gs = attach.front_cull(gq, gn, gs, view_dir=(0.0, -1.0, 0.0))

                q_all = np.vstack([bq, gq])
                keep = mask_of(zbuffer_cull(q_all, pixel=0.003, keep_depth=0.02), len(q_all))
                kb, kt = keep[:len(bq)], keep[len(bq):]
                ng = int(kt.sum())

                shadow = keep_b0 & ~kb
                sig = sig_body.copy()
                if shadow.any():
                    sig -= fwd(bq[shadow], sb[shadow])
                if ng:
                    sig += fwd(gq[kt], gs[kt])

                mag = recon(sig, grid)
                pk = float(mag.max())
                db = (20 * np.log10(np.maximum(mag.max(1), pk * 1e-4) / pk)).astype(np.float32)

                if ng:
                    gk = gq[kt]
                    gx, gy, gz = vox_idx(gk, grid)
                    gun_vox = np.unique(np.column_stack([gx, gy, gz]), axis=0).astype(np.int16)
                    gun_db = 20 * np.log10(max(float(np.percentile(mag[gx, gy, gz], 90)),
                                               pk * 1e-5) / pk)
                    seat = gk.mean(0)
                    aabb = np.vstack([gk.min(0), gk.max(0)])
                    vis = ng / n_seated
                    obliq = float(np.degrees(np.arccos(np.clip(np.abs(-gn[kt][:, 1]).mean(), 0, 1))))
                else:
                    gun_vox = np.zeros((0, 3), np.int16)
                    gun_db = vis = obliq = np.nan
                    seat = np.full(3, np.nan)
                    aabb = np.full((2, 3), np.nan)

                np.savez_compressed(
                    f"{OUT}/pb_{bid}_{stance}_{tag}_{site}.npz", site=site,
                    face=pl["face"], bary=np.array(pl["bary"]), flip=bool(pl["flip"]),
                    han_db=db, mag3d=mag.astype(np.float32), han_peak=pk,
                    seat_xyz=seat, gun_vox=gun_vox, gun_aabb=aabb,
                    n_seated=n_seated, n_visible=ng, visibility=vis, obliquity_deg=obliq,
                    han_gun_db=gun_db, **common)
                print(f"    {bid}/{stance}/{tag}/{site:<15} vis {vis:.3f} gun {gun_db:+6.1f} dB "
                      f"nvox {len(gun_vox):,} ({time.time()-t1:.0f}s)", flush=True)

            del bq, sb, sig_body
        del bq_raw

print(f"[{datetime.datetime.now():%H:%M}] PROBE done -> {OUT}", flush=True)
