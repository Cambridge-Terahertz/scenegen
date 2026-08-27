#!/usr/bin/env python3
"""Extract per-scan METADATA only from the ds6 pilot (no pixels) -> one npz table.
This is what the position-only gate is scored on: if site/location alone can predict the
threat label, the shortcut survived."""
import numpy as np, glob, os, sys
import sys as _sys; _sys.path.insert(0, "/home/jonah/probe")
import frame  # world-frame loader: normalises legacy files on read

from multiprocessing import Pool

DATA = "/mnt/shared_drive/Jonah/synthetic_data/dataset6_2026-08-25"
OUT  = os.path.expanduser("~/probe/ds6_meta.npz")
KEYS = ("body","stance","site","obj_name","is_threat","plausible","view_deg",
        "placement_idx","visibility","obliquity_deg","roll_deg","gap_m","flip",
        "han_contrast_db","n_visible")

def proc(f):
    try:
        z = frame.load_scan(f)          # legacy files arrive converted to the world frame
        r = {k: (z[k].item() if getattr(z[k], "shape", None) == () else z[k]) for k in KEYS if k in z}
        ov = z["obj_vox"]
        if len(ov):
            lo = ov.min(0); hi = ov.max(0)
            # world frame: the image plane is (x, y); range is z and is NOT part of the bbox
            r["bb"] = np.array([(lo[0]+hi[0])/2, (lo[1]+hi[1])/2,
                                hi[0]-lo[0]+1, hi[1]-lo[1]+1], np.float32)
            r["nvox"] = float(len(ov))
        else:
            r["bb"] = np.full(4, np.nan, np.float32); r["nvox"] = 0.0
        return r
    except Exception as e:
        return {"err": f"{os.path.basename(f)}: {e}"}

if __name__ == "__main__":
    fs = sorted(glob.glob(os.path.join(DATA, "**", "*.npz"), recursive=True))
    print(f"{len(fs)} files", flush=True)
    with Pool(24) as p:
        rows = p.map(proc, fs, chunksize=64)
    errs = [r["err"] for r in rows if "err" in r]
    rows = [r for r in rows if "err" not in r]
    print(f"ok={len(rows)} err={len(errs)}", flush=True)
    for e in errs[:5]: print("  ", e)
    out = {}
    for k in KEYS:
        vals = [r.get(k) for r in rows]
        if all(v is None for v in vals): continue
        out[k] = np.array(vals)
    out["bb"]   = np.stack([r["bb"] for r in rows])
    out["nvox"] = np.array([r["nvox"] for r in rows])
    np.savez_compressed(OUT, **out)
    print("wrote", OUT, {k: out[k].shape for k in out}, flush=True)
