"""4-system / 2-pole coherent portal, threat-site sweep, on the FAST validated pilot generator.

Reuses ~/scenegen/pilot_aperture_driver.py VERBATIM for everything up to its dataset loop
(imports, SMPL-X body at lam/8, object library, attach/front_cull/zbuffer, world frame, tapered
ROI crop) by exec'ing its setup section; the per-channel forward + recon is driven here.

Validated fast settings (aperture pilot 2026-09-30, corr 0.988-0.997 vs full body / lam/2 / 256f):
  ROI +/-25 cm with 3 cm raised-cosine edge taper, element pitch 2.5 mm, 64 freqs.

World frame (frame.py): +x person-LEFT, +y up, +z into scene, array plane z = 0, body at 2 m.
Per look direction: 4 reflectarrays (7 cm grids, rect aperture) at the pole corners
(x = +/-0.50 m, heights 0.75/1.65 m) and 4 probes. 16 channels = (array a as receive grid,
probe r as the transmitter point) -- identical to array-TX/probe-RX by reciprocity.
raforward is isotropic (no element pattern), so array toe-in has no effect and is not modelled.

Configs per (site, object), identical scene/grid/physics:
  dk_single: ONE DK unit at boresight (array x=0 h=1.20 m, own probe at DK offset) = today
  16coh   : coherent sum of the 16 complex fields
  16incoh : sum of the 16 field magnitudes
Metric (like-for-like): same object footprint imaged WITH and WITHOUT the object.

Env: PROBES=opt|nominal|onpanel  SITES=...  OBJS=...  VX_MM  HALFW  OUT
"""
import os, sys, time, json
import numpy as np

# ---- configure the pilot driver for the validated fast settings, then exec its setup ----
os.environ.update(DS7_D="0.07", DS7_PITCH="0.0025", DS7_NFREQ=os.environ.get("NF","128"), DS7_ROI="0.25",
                  DS7_ORIENTS="front", DS7_NBODIES="1", DS7_APER="rect",
                  DS7_OUT=os.path.expanduser("~/fourpole_tmp"))
os.environ.setdefault("DS7_ARMS", "up")
DRV = os.path.expanduser("~/scenegen/pilot_aperture_driver.py")
src = open(DRV).read()
setup = src.split("\nnscan = 0\n")[0]
__file__ = DRV                                    # driver resolves paths from its own location
exec(compile(setup, DRV, "exec"))                 # defines rf, Scene, backproject_points, etc.

from scipy import ndimage as ndi

OUT4 = os.environ.get("OUT", os.path.expanduser("~/fourpole_out")); os.makedirs(OUT4, exist_ok=True)
SITES4 = os.environ.get("SITES", "wb_front_mid,wb_front_l,wb_front_r,belly,chest,thigh_front_r").split(",")
OBJS4 = os.environ.get("OBJS", "gun1_glock17,phone_modern").split(",")
PROBES = os.environ.get("PROBES", "opt")
VX4 = float(os.environ.get("VX_MM", "1.0")) / 1000.0
HALFW4 = float(os.environ.get("HALFW", "0.12"))
ZOFF = np.arange(-0.03, 0.0301, 0.0025)          # envelope shell about the front surface

# hardware in ABSOLUTE floor heights (m); x = +person-left
PANELS = [(-0.5, 0.75), (-0.5, 1.65), (0.5, 0.75), (0.5, 1.65)]
PROBE_SETS = {
    "opt":     [(-0.022, 1.408), (0.059, 1.537), (-0.173, 1.347), (0.117, 1.366)],  # probe_opt.py
    "nominal": [(-0.25, 0.975), (-0.25, 1.425), (0.25, 0.975), (0.25, 1.425)],
    "onpanel": [(-0.5, 0.75), (-0.5, 1.65), (0.5, 0.75), (0.5, 1.65)],
}


def build_hw(floor_to_world):
    """grids (list of (nrx,3)) for the 4 arrays and (1,3) tx points for the 4 probes, world frame."""
    grids = [np.column_stack([RX[:, 0] + x, RX[:, 1] + floor_to_world(h), RX[:, 2]]) for x, h in PANELS]
    probes = [np.array([[x, floor_to_world(h), 0.0]]) for x, h in PROBE_SETS[PROBES]]
    return grids, probes


def fwd_ch(q, s, tx, rx):
    sig = np.zeros((NFREQ, len(rx)), np.complex64)
    for c0 in range(0, len(q), 2_000_000):
        sig += rf.forward(Scene(q=q[c0:c0+2_000_000], s=s[c0:c0+2_000_000], tx=tx, rx=rx, k=KV),
                          precision="fp32")
    return sig


def recon_ch(sig, tx, rx, pts):
    return _to_np(backproject_points(sig * _APER_W[None, :], tx=tx, rx=rx, f=FREQ, points=pts,
                                     freq_window=FREQ_WINDOW))


# ---- one body (pilot body 0), front view ----
shp = SHAPES[0]; bid = bid_of(shp)
model = bodylib.get_model(shp["gender"]); VC = bodylib.canonical_verts(model)
Vp, F = bodylib.posed_verts(model, betas=np.array(shp["betas"]), body_pose=POSE)
m = trimesh.Trimesh(vertices=Vp, faces=F, process=False); m.fix_normals()
yc = 0.5 * (Vp[:, 1].min() + Vp[:, 1].max())
f2w = lambda h: h - (yc - Vp[:, 1].min())          # absolute floor height -> world y
Vs, Fs, _ = subdivide_front(m, 0.5 * LAM)
bq_raw, _, _ = sample_mesh(Vs, Fs, 0.125 * LAM, front_cull=False)
stance = "front"
bq = to_world(bq_raw.astype(np.float64), stance, yc)
sb = np.full(len(bq), S_SKIN)
keep_b0 = mask_of(zbuffer_cull(bq, pixel=0.003, depth_axis=2, keep_depth=0.02), len(bq))
GRIDS, PROBES_W = build_hw(f2w)
print(f"[4pole] body {bid}  {len(bq_raw):,} pts  | probes={PROBES}  | {len(GRIDS)} arrays x "
      f"{len(PROBES_W)} probes, {len(RX)} el/array, pitch {PITCH*1e3:.1f} mm, NF {NFREQ}", flush=True)
# BASELINE = one DK unit at boresight: 7 cm array at x=0, h=1.20 m, own probe at the DK offset
DK_GRID = np.column_stack([RX[:, 0], RX[:, 1] + f2w(1.20), RX[:, 2]])
DK_TX = np.array([[0.15875, f2w(1.20) + 0.05715, 0.0]])

rows = []
for site in SITES4:
    t0 = time.time()
    cx = site_ctx(site, model, Vp, F, stance, yc, bq, sb, keep_b0)   # tapered ROI (validated)
    roi, sbw, ctr = cx["roi"], cx["sbw"], cx["centre"]
    for obj_name in OBJS4:
        obj, is_threat = OBJECTS[obj_name]
        pl = anatomy.sample_placement(model, site, np.random.default_rng(0), jitter=False)
        pl.update(roll=ROLL_OF[obj_name], standoff=GAP, flip=False)
        gq, gn, gs = attach.attach(obj, Vp, VC, F, pl, target_nn=0.125 * LAM, vert_normals=m.vertex_normals)
        gq = to_world(gq, stance, yc); gn = to_world(gn, stance, yc, is_dir=True)
        gq, gn, gs = attach.front_cull(gq, gn, gs, view_dir=(0.0, 0.0, -1.0))
        q_all = np.vstack([bq, gq])
        keep = mask_of(zbuffer_cull(q_all, pixel=0.003, depth_axis=2, keep_depth=0.02), len(q_all))
        kb, kt = keep[:len(bq)], keep[len(bq):]
        if OBJ_ZBUF and kt.any():
            _i = np.flatnonzero(kt)
            _k2 = mask_of(zbuffer_cull(np.ascontiguousarray(gq[_i]), pixel=OBJ_ZPIX, depth_axis=2,
                                       keep_depth=OBJ_ZDEPTH), len(_i))
            kt = np.zeros_like(kt); kt[_i[_k2]] = True
        if not kt.any():
            print(f"  {site}/{obj_name}: object not visible, skip", flush=True); continue
        shadow = roi & ~kb
        go = gq[kt]
        cxo, cyo = go[:, 0].mean(), go[:, 1].mean()
        xs = np.arange(cxo - HALFW4, cxo + HALFW4, VX4); ys = np.arange(cyo - HALFW4, cyo + HALFW4, VX4)
        # front-surface depth map (body ROI + object) on the image grid -> shell points
        surf = np.vstack([bq[roi & kb], go])
        inb = ((surf[:, 0] >= xs[0]) & (surf[:, 0] < xs[-1] + VX4) &
               (surf[:, 1] >= ys[0]) & (surf[:, 1] < ys[-1] + VX4))   # DROP out-of-patch pts (no clip)
        surf = surf[inb]
        ix = np.clip(np.searchsorted(xs, surf[:, 0], side="right") - 1, 0, len(xs) - 1)
        iy = np.clip(np.searchsorted(ys, surf[:, 1], side="right") - 1, 0, len(ys) - 1)
        dep = np.full((len(xs), len(ys)), np.inf); np.minimum.at(dep, (ix, iy), surf[:, 2])
        ii, jj = np.where(np.isfinite(dep))
        nd = len(ZOFF)
        zspan = dep[ii, jj].max() - dep[ii, jj].min() + (ZOFF[-1] - ZOFF[0])
        print(f"  {site}/{obj_name}: shell depth span {zspan*100:.1f} cm (window {C0/(FREQ[1]-FREQ[0])*100:.0f} cm path)", flush=True)
        pts = np.ascontiguousarray(np.column_stack([
            np.repeat(xs[ii], nd), np.repeat(ys[jj], nd), (dep[ii, jj][:, None] + ZOFF[None, :]).ravel()]))

        fields = {"with": {}, "body": {}}
        dk = {}
        sb_ = fwd_ch(bq[roi], sbw[roi], DK_TX, DK_GRID); so_ = sb_.copy()
        if shadow.any():
            so_ -= fwd_ch(bq[shadow], sbw[shadow], DK_TX, DK_GRID)
        so_ += fwd_ch(go, gs[kt], DK_TX, DK_GRID)
        dk["with"] = recon_ch(so_, DK_TX, DK_GRID, pts); dk["body"] = recon_ch(sb_, DK_TX, DK_GRID, pts)
        for a, rxg in enumerate(GRIDS):
            for r, tx in enumerate(PROBES_W):
                sig_b = fwd_ch(bq[roi], sbw[roi], tx, rxg)
                sig_o = sig_b.copy()
                if shadow.any():
                    sig_o -= fwd_ch(bq[shadow], sbw[shadow], tx, rxg)
                sig_o += fwd_ch(go, gs[kt], tx, rxg)
                fields["with"][(a, r)] = recon_ch(sig_o, tx, rxg, pts)
                fields["body"][(a, r)] = recon_ch(sig_b, tx, rxg, pts)

        def img(v):
            out = np.zeros((len(xs), len(ys)), np.float32)
            out[ii, jj] = np.abs(v).reshape(-1, nd).max(1); return out
        res = {}
        for key in ("with", "body"):
            fs = fields[key]
            res[("dk_single", key)] = img(dk[key])
            res[("16coh", key)] = img(sum(fs.values()))
            res[("16incoh", key)] = img(sum(np.abs(v) for v in fs.values()))
        # object footprint mask
        dx = VX4
        g, _, _ = np.histogram2d(go[:, 0], go[:, 1], bins=[np.r_[xs - dx/2, xs[-1] + dx/2],
                                                           np.r_[ys - dx/2, ys[-1] + dx/2]])
        G = ndi.binary_fill_holes(ndi.binary_closing(g > 0, np.ones((3, 3)), iterations=2))
        dbf = lambda v: float(20 * np.log10(v))
        line = f"  {site:<14} {obj_name:<13}"
        for cfg in ("dk_single", "16coh", "16incoh"):
            A, B = res[(cfg, "with")], res[(cfg, "body")]
            ok = G & (A > 0) & (B > 0)
            row = dict(site=site, obj=obj_name, threat=bool(is_threat), config=cfg, probes=PROBES,
                       vox_mm=VX4 * 1e3, obj_px=int(ok.sum()),
                       add_median_dB=dbf(np.median(A[ok]) / np.median(B[ok])),
                       add_p95_dB=dbf(np.percentile(A[ok], 95) / np.percentile(B[ok], 95)),
                       add_peak_dB=dbf(A[ok].max() / B[ok].max()),
                       obj_vs_bodypeak_dB=dbf(A[ok].max() / A.max()))
            rows.append(row)
            line += f" | {cfg} med {row['add_median_dB']:+5.1f} p95 {row['add_p95_dB']:+5.1f} vsPk {row['obj_vs_bodypeak_dB']:+5.1f}"
        print(line + f"  [{time.time()-t0:.0f}s]", flush=True)
        np.savez_compressed(f"{OUT4}/{PROBES}_{site}_{obj_name}.npz", xs=xs, ys=ys, mask=G,
                            **{f"{c}_{k}": v for (c, k), v in res.items()})
        json.dump(rows, open(f"{OUT4}/sweep_{PROBES}.json", "w"), indent=1)
print("done", flush=True)
