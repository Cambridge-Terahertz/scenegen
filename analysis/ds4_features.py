#!/usr/bin/env python
"""dataset4 detection features (plan section 7 step 5).

Per scan, three CLEARLY SEPARATED feature groups:

  BLIND  (deployable -- computed from the RF image alone, available at test time):
     hb_*   : han_db brightness stats over the body silhouette
     bl*    : blind blob shape on han_db at several dB thresholds (no location prior)
     rng*   : range structure from field_gated (the +-5 cm shell max(1) collapses)

  PROBE  (label-assisted -- uses the KNOWN object box; eval-only, NOT a detector feature):
     ow_*   : object-window blob shape (image-derived shape, but located by the label)
     n_obj_vox, contrast_db : truth ceilings

Leakage fields (obj_vox/aabb, seat_xyz, visibility, obliquity, han_obj/contrast, depth_map)
are used ONLY to build PROBE features / eval masks, never as BLIND features.

Writes ds4_feat.csv .
"""
import os, glob, csv, sys
import numpy as np
from skimage.measure import label, regionprops

NAS = os.environ.get("DS4_OUT", "/Volumes/Shared_Drive/Jonah/synthetic_data/dataset4_2026-07-21")
OUT = os.environ.get("FEAT_OUT", "/Users/jonah/.claude/jobs/dab08409/tmp/ds4_feat.csv")
BIN_CM = 0.5            # field_gated range bin = 5 mm
CENTER = 10            # surface bin

def blob_feats(imgdb, thr_db, min_area=3):
    """Blobs on a dB image at (max - thr_db). -> dict of aggregate shape features."""
    m = imgdb >= (np.nanmax(imgdb) - thr_db)
    lab = label(m)
    props = [p for p in regionprops(lab) if p.area >= min_area]
    if not props:
        return dict(n=0, maxarea=0.0, maxasp=0.0, toparea=0.0, topasp=0.0, topfill=0.0)
    areas = np.array([p.area for p in props], float)
    asp = np.array([p.axis_major_length / (p.axis_minor_length + 1e-6) for p in props])
    top = props[int(areas.argmax())]
    return dict(n=len(props), maxarea=float(areas.max()), maxasp=float(asp.max()),
                toparea=float(top.area),
                topasp=float(top.axis_major_length / (top.axis_minor_length + 1e-6)),
                topfill=float(top.extent))

def main():
    files = sorted(glob.glob(f"{NAS}/ds4_*.npz"))
    print(f"{len(files)} files -> {OUT}", flush=True)
    rows = []
    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        xs, zs = np.asarray(d["xs"]), np.asarray(d["zs"])
        nx, nz = len(xs), len(zs)
        hb = np.asarray(d["han_db"], float)               # (nx,nz) absolute dB
        dm = np.asarray(d["depth_map"], float)
        sil = np.isfinite(dm)                             # silhouette mask
        r = dict(file=os.path.basename(f), body=str(d["body"]), stance=str(d["stance"]),
                 obj_name=str(d["obj_name"]), site=str(d["site"]),
                 is_threat=bool(d["is_threat"]))

        # ---- BLIND: brightness over the silhouette ----
        v = hb[sil]
        r["hb_max"] = float(v.max()); r["hb_p50"] = float(np.percentile(v, 50))
        r["hb_p90"] = float(np.percentile(v, 90)); r["hb_p99"] = float(np.percentile(v, 99))
        r["hb_dyn"] = r["hb_max"] - r["hb_p50"]

        # ---- BLIND: blind blob shape on han_db at several dB depths ----
        hbm = np.where(sil, hb, -np.inf)
        for thr in (6, 10, 14):
            b = blob_feats(hbm, thr)
            for k, val in b.items():
                r[f"bl{thr}_{k}"] = val

        # ---- BLIND: range structure from field_gated (the collapsed axis) ----
        fg = np.asarray(d["field_gated"]); gc = np.asarray(d["gate_cells"])
        mag = np.abs(fg).astype(float)                    # (Ngate,21)
        ix, iz = gc // nz, gc % nz
        bins = np.arange(fg.shape[1])
        w = mag / (mag.sum(1, keepdims=True) + 1e-12)
        centroid = (w * bins).sum(1)
        spread_cm = np.sqrt((w * (bins[None] - centroid[:, None]) ** 2).sum(1)) * BIN_CM
        peak = mag.argmax(1)
        offset_cm = np.abs(peak - CENTER) * BIN_CM        # |proud/deep| distance from surface
        proud_frac = mag[:, :CENTER].sum(1) / (mag.sum(1) + 1e-12)
        r["rng_maxspread"] = float(spread_cm.max()); r["rng_p99spread"] = float(np.percentile(spread_cm, 99))
        r["rng_maxoffset"] = float(offset_cm.max()); r["rng_p99offset"] = float(np.percentile(offset_cm, 99))
        r["rng_proud_max"] = float(proud_frac.max())
        r["rng_area_off1cm"] = float((offset_cm > 1.0).sum())     # cells returning >1cm off-surface
        r["rng_area_spr1cm"] = float((spread_cm > 1.0).sum())
        # blind blob on the range-offset map (3D anomaly, no location prior)
        offmap = np.full((nx, nz), 0.0); offmap[ix, iz] = offset_cm
        ob = blob_feats(np.where(offmap > 0, offmap, -np.inf), thr_db=offmap.max() * 0.5 if offmap.max() > 0 else 1,
                        min_area=2) if offmap.max() > 0.8 else dict(n=0, maxarea=0.0, maxasp=0.0)
        r["rng_blob_n"] = ob["n"]; r["rng_blob_maxarea"] = ob["maxarea"]; r["rng_blob_maxasp"] = ob["maxasp"]

        # ---- PROBE: object-window (label-located) image shape + range ----
        ab = np.asarray(d["obj_aabb"], float); pad = 0.02
        mx = (xs >= ab[0, 0] - pad) & (xs <= ab[1, 0] + pad)
        mz = (zs >= ab[0, 2] - pad) & (zs <= ab[1, 2] + pad)
        if mx.any() and mz.any():
            sub = hbm[np.ix_(mx, mz)]
            ow = blob_feats(sub, thr_db=6, min_area=2)
            r["ow_area"] = ow["toparea"]; r["ow_asp"] = ow["topasp"]; r["ow_fill"] = ow["topfill"]
            inwin = mx[ix] & mz[iz]
            r["ow_rng_offset"] = float(offset_cm[inwin].max()) if inwin.any() else 0.0
            r["ow_rng_spread"] = float(spread_cm[inwin].mean()) if inwin.any() else 0.0
            # object range THICKNESS: window-summed range profile, extent above half-max (cm)
            if inwin.any():
                prof = mag[inwin].sum(0)                   # (21,) aggregate range profile
                half = prof >= 0.5 * prof.max()
                r["ow_rng_thick"] = float(half.sum()) * BIN_CM
                r["ow_rng_peakoff"] = float(abs(int(prof.argmax()) - CENTER)) * BIN_CM
            else:
                r["ow_rng_thick"] = r["ow_rng_peakoff"] = 0.0
        else:
            r["ow_area"] = r["ow_asp"] = r["ow_fill"] = r["ow_rng_offset"] = r["ow_rng_spread"] = 0.0
            r["ow_rng_thick"] = r["ow_rng_peakoff"] = 0.0
        # ---- truth ceilings (eval only) ----
        r["n_obj_vox"] = float(d["obj_vox"].shape[0]) if d["obj_vox"].size else 0.0
        r["contrast_db"] = float(d["han_contrast_db"])
        r["visibility"] = float(d["visibility"])

        rows.append(r)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

    keys = list(rows[0].keys())
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"DONE -> {OUT}  ({len(rows)} rows, {len(keys)} cols)", flush=True)

if __name__ == "__main__":
    main()
