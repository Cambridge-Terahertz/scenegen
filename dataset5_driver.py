"""Dataset 5: expanded threat+benign object set, same physics/save format as dataset4.

WHAT CHANGED vs dataset4 (dataset4_driver.py):
  OBJECT SET       17 objects from objects_v5 (see objects_v5/manifest.json) via placement_config:
                   threats  = gun1_glock17 (the SAME two-material Glock as dataset4, the anchor),
                              gun2_glock19, gun3_hellcat, gun4_mp9, gun5_derringer,
                              knife_{switchblade,folding,corkscrew,hunting,dielectric}
                   benign   = phone_nokia, phone_modern, wallet, badge, keys, necklace, belt
  PLACEMENTS       threats -> ALL 45 sites (36 leaf); benign -> their reviewed OK sites
                   (confuser_site_map.json). See placement_config.py.
  PER-OBJECT ROLL  each object seats with its own roll offset (objects_v5 manifest), instead of
                   dataset4's global ROLL=0.
  MATERIALS        gun1 = two-material (steel slide + polymer frame, threats._glock_parts, as
                   dataset4). Every other object = single material from the manifest (user
                   decision 2026-07-24: "rest steel"; dielectric knife = ceramic).

UNCHANGED from dataset4 (deliberately -- must stay comparable): 2.0 m standoff, taylor(sll=40)
both tapers, +/-5 cm complex range gate, 1 cm cross-range grid, 5 mm range step, body-blocked
delta forward, han_db definition, and the 3D save (field_gated shell + gate_cells lookup).

STILL NOT INCLUDED: roughness + noise floor (plan sec.7 step 2). SPECULAR dataset.

  DS5_OUT=/path DS5_NBODIES=8 ~/miniconda3/envs/ra/bin/python dataset5_driver.py
"""
import datetime, json, os, sys, time
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir("/Users/jonah"):
    os.environ.setdefault("SMPLX_MODELS", os.path.expanduser("~/smplx_models"))
    os.environ.setdefault("AMASS_ROOT", os.path.expanduser("~/amass"))
os.environ.setdefault("RA_PHASE_RECURSION", "1")
for p in (os.path.expanduser("~/walkthrough"),
          os.path.expanduser("~/Documents/code/model_gen/python"),
          "/Users/jonah/Documents/scratch_results/threat_dataset"):
    if os.path.isdir(p):
        sys.path.insert(0, p)
sys.path.insert(0, HERE)

from mesh_sample import sample_mesh                      # noqa: E402
from bodies_smplx import subdivide_front                 # noqa: E402
import anatomy, attach, threats, body as bodylib         # noqa: E402
import placement_config as pc                            # noqa: E402  (registers the 45 sites)
import raforward as rf                                   # noqa: E402
from raforward import Scene, C0                          # noqa: E402
from recon import ReconGrid, backproject                 # noqa: E402
from bodies import zbuffer_cull                          # noqa: E402
from stance import screening_pose                        # noqa: E402

if not rf._HAS_CUPY:
    sys.exit("FATAL: no CuPy -- use ~/miniconda3/envs/ra/bin/python on a GPU host (bose/tesla).")

DS5_NAME = "dataset5_2026-07-24"
OUT = os.environ.get("DS5_OUT", os.path.expanduser(f"~/{DS5_NAME}"))
os.makedirs(OUT, exist_ok=True)
TD = "/Users/jonah/Documents/scratch_results/threat_dataset" if os.path.isdir(
    "/Users/jonah/Documents/scratch_results/threat_dataset") else HERE
V5 = next(p for p in (f"{HERE}/objects_v5", f"{TD}/objects_v5",
                      os.path.expanduser("~/probe/objects_v5")) if os.path.isdir(p))
GLOCK_STL = next(p for p in (f"{HERE}/glock17_ext.stl", f"{TD}/glock17_ext.stl",
                             os.path.expanduser("~/walkthrough/glock17_ext.stl"),
                             os.path.expanduser("~/threat_dataset/glock17_ext.stl"))
                 if os.path.exists(p))
PHONE = next(p for p in (os.path.join(os.path.dirname(GLOCK_STL), "phone_hull.stl"),
                         os.path.expanduser("~/walkthrough/phone_hull.stl"),
                         os.path.expanduser("~/threat_dataset/phone_hull.stl"))
             if os.path.exists(p))

LAM = C0 / 290e9
NFREQ = 256
STANDOFF = float(os.environ.get("DS5_STANDOFF", "2.0"))
GATE_T = float(os.environ.get("DS5_GATE", "0.05"))
SIG_EVERY = int(os.environ.get("DS5_SIG_EVERY", "128"))
GAP = 0.012
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
WINDOW = os.environ.get("DS5_WINDOW", "taylor")
if WINDOW == "taylor":
    from scipy.signal.windows import taylor as _taylor
    APER = _taylor(130, nbar=5, sll=40, norm=False)
elif WINDOW == "hann":
    APER = np.hanning(130)
else:
    sys.exit(f"DS5_WINDOW must be 'taylor' or 'hann', got {WINDOW!r}")
POSE = screening_pose()
S_SKIN = threats.MATERIALS["skin"][0]
SHAPES = json.load(open(os.environ.get("DS5_SHAPES", f"{HERE}/shapes_10x10.json")))
_NB = int(os.environ.get("DS5_NBODIES", "0"))
if _NB:
    SHAPES = SHAPES[:_NB]

# ---------------------------------------------------------------- object catalogue ----------
# gun1 = the dataset4 two-material Glock (build_library does the steel/polymer split). Everything
# else = an objects_v5 mesh (already canonical + real-scale) wrapped as a single-material Threat.
OBJECTS = {"gun1_glock17": (threats.build_library(glock_stl=GLOCK_STL, phone_stl=PHONE)["pistol"], True)}
for name, o in pc.OBJECTS.items():
    if name == "gun1_glock17":
        continue
    mesh = trimesh.load(f"{V5}/{name}.obj", process=False)
    OBJECTS[name] = (threats.Threat(name, o["kind"], o["material"], mesh=mesh, conformal=False),
                     o["kind"] == "threat")
ROLL_OF = {name: float(o["roll_deg"]) for name, o in pc.OBJECTS.items()}
ROLL_OF["gun1_glock17"] = 0.0

BACK_SITES = {"small_of_back", "upper_back", "wb_back_mid", "wb_back_l", "wb_back_r",
              "thigh_back_l", "thigh_back_r", "shin_back_l", "shin_back_r"}
def stance_of(site):
    return "back" if site in BACK_SITES else "front"

def placements(stance):
    """[(obj_name, site)] for a stance -- baseline first so it lands even if a run is cut short."""
    out = [("none", "nothreat")]
    for name, sites in pc.PLACEMENT.items():
        out += [(name, s) for s in sites if stance_of(s) == stance]
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
    m = np.zeros(n, bool); m[np.asarray(idx).ravel()] = True; return m

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
    """Range-gate the cube about the per-cell surface depth -> (n_cells, KGATE) complex64 +
    gate_cells lookup (flat (ix,iz) index; iy implicit from dmap + ys[0] + DY). Lossless subset."""
    cells = np.flatnonzero(np.isfinite(dmap.ravel()))
    ix, iz = np.unravel_index(cells, dmap.shape)
    iy0 = np.round((dmap.ravel()[cells] - grid.ys[0]) / DY).astype(np.int32)
    off = np.arange(KGATE, dtype=np.int32) - KGATE // 2
    iy = np.clip(iy0[:, None] + off[None, :], 0, len(grid.ys) - 1)
    return field[ix[:, None], iy, iz[:, None]].astype(np.complex64), cells.astype(np.int32)


N_PER_BODY = sum(len(placements(st)) for st in ("front", "back"))
N_THREAT = sum(1 for st in ("front", "back") for o, s in placements(st)
               if o != "none" and OBJECTS[o][1])
print(f"[{datetime.datetime.now():%H:%M}] DATASET5: {len(SHAPES)} bodies x {N_PER_BODY} placements "
      f"= {len(SHAPES)*N_PER_BODY} scans ({len(SHAPES)*N_THREAT} threat / "
      f"{len(SHAPES)*(N_PER_BODY-N_THREAT)} no-threat)", flush=True)
print(f"  objects: " + ", ".join(f"{n}{'*' if t else ''}" for n, (_, t) in OBJECTS.items())
      + "   (* = threat)", flush=True)
print(f"  standoff={STANDOFF} m  window={WINDOW}  gate=+/-{GATE_T*100:.0f} cm (K={KGATE})  "
      f"sig every {SIG_EVERY}  GPU={rf._HAS_CUPY} -> {OUT}", flush=True)

nscan = 0
t_start = time.time()
for bi, shp in enumerate(SHAPES):
    bid = bid_of(shp)
    if all(os.path.exists(f"{OUT}/ds5_{bid}_{st}_{o}_{s}.npz")
           for st in ("front", "back") for o, s in placements(st)):
        print(f"  skip {bid} (done)", flush=True); nscan += N_PER_BODY; continue

    t0 = time.time()
    betas = np.array(shp["betas"])
    model = bodylib.get_model(shp["gender"])
    VC = bodylib.canonical_verts(model)
    Vp, F = bodylib.posed_verts(model, betas=betas, body_pose=POSE)
    m = trimesh.Trimesh(vertices=Vp, faces=F, process=False); m.fix_normals()
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

        for obj_name, site in placements(stance):
            t1 = time.time()
            f_out = f"{OUT}/ds5_{bid}_{stance}_{obj_name}_{site}.npz"
            if os.path.exists(f_out):
                nscan += 1; continue

            if obj_name == "none":
                sig, ng, n_seated, pl, is_threat = sig_body, 0, 0, None, False
                gq = kt = gn = None
            else:
                obj, is_threat = OBJECTS[obj_name]
                pl = anatomy.sample_placement(model, site, np.random.default_rng(0), jitter=False)
                pl.update(roll=ROLL_OF[obj_name], standoff=GAP, flip=False)
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
                shadow = keep_b0 & ~kb
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
                gy = np.clip(np.round((gk[:,1]-grid.ys[0])/DY).astype(np.int32), 0, len(grid.ys)-1)
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
            print(f"    {bid}/{stance}/{obj_name:<16}/{site:<16} "
                  f"{'THREAT' if is_threat else 'benign':<6} vis "
                  f"{rec['visibility'] if ng else 0:.3f} contrast "
                  f"{rec['han_contrast_db']:+6.1f} dB  ({time.time()-t1:.0f}s)"
                  + ("  +sig" if "sig" in rec else ""), flush=True)
        del bq, sb, sig_body
    del bq_raw, Vp, F, m, Vs, Fs

print(f"[{datetime.datetime.now():%H:%M}] DATASET5 done: {nscan} scans in "
      f"{(time.time()-t_start)/60:.0f} min -> {OUT}", flush=True)
