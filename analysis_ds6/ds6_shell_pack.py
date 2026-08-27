#!/usr/bin/env python3
"""Pack the ds6 range SHELL for the 2.5D experiment.

`han_db` is a max-projection over range: the depth axis is discarded. `field_gated` keeps 21
complex samples at 5 mm spacing around each surface cell, so an object sitting proud of the skin
has a range signature the projection destroys. This packs that shell so a 2.5D model can use it.

Both arms are derived from the SAME source here -- the 21-bin stack and a 1-bin max-projection of
that same stack -- so "does range help" is a controlled comparison rather than a comparison
confounded by preprocessing. (The earlier 0.614 used the driver's `han_db`, a projection of the
FULL cube rather than of the shell, so it is not the right control.)

Magnitudes are block-averaged in LINEAR amplitude then converted to dB; averaging dB directly
would bias toward the dim bins.
"""
import numpy as np, glob, os, sys
import sys as _sys; _sys.path.insert(0, "/home/jonah/probe")
import frame  # world-frame loader: normalises legacy files on read

from multiprocessing import Pool

DATA = "/mnt/shared_drive/Jonah/synthetic_data/dataset6_2026-08-25"
OUT  = os.path.expanduser("~/probe/ds6_shell.npz")
NR, NC, KG = 180, 210, 21
H, W = NR//2, NC//2                      # 90 x 105

def proc(f):
    try:
        z = frame.load_scan(f)          # world frame regardless of vintage
        fg = z["field_gated"]; gc = z["gate_cells"]
        nx, nz = len(z["xs"]), len(z["ys"])   # image plane is (x, y); gate_cells indexes it
        body = str(z["body"]); view = str(z["stance"]); obj = str(z["obj_name"])
        thr = bool(z["is_threat"])
        vol = np.zeros((nx, nz, KG), np.float32)
        ix, iy = np.unravel_index(np.asarray(gc), (nx, nz))
        vol[ix, iy, :] = np.abs(fg).astype(np.float32)
        vol = vol[:NR, :NC].reshape(H, 2, W, 2, KG).mean(axis=(1, 3))   # linear average
        pk = vol.max()
        if not np.isfinite(pk) or pk <= 0:
            return (os.path.basename(f), body, view, obj, thr, None)
        db = 20*np.log10(np.maximum(vol, pk*1e-3)/pk)                  # dB rel per-scan peak
        q = np.clip((db + 60.0)/60.0, 0, 1)
        return (os.path.basename(f), body, view, obj, thr,
                (q*255).astype(np.uint8).transpose(2, 0, 1))            # (21, H, W)
    except Exception as e:
        return ("ERR", str(e), None, None, None, None)

if __name__ == "__main__":
    fs = sorted(glob.glob(os.path.join(DATA, "**", "*.npz"), recursive=True))
    print(f"{len(fs)} files -> ({KG}, {H}, {W})", flush=True)
    with Pool(24) as p:
        rows = p.map(proc, fs, chunksize=32)
    bad = [r for r in rows if r[0] == "ERR"]
    print(f"errors: {len(bad)}", flush=True)
    for b in bad[:5]: print("  ", b[1])
    rows = [r for r in rows if r[0] != "ERR"]
    empty = sum(1 for r in rows if r[5] is None)
    print(f"empty shells: {empty}", flush=True)

    name = np.array([r[0] for r in rows]); body = np.array([r[1] for r in rows])
    view = np.array([r[2] for r in rows]); obj  = np.array([r[3] for r in rows])
    y    = np.array([r[4] for r in rows], bool)
    Z = np.zeros((len(rows), KG, H, W), np.uint8)
    for i, r in enumerate(rows):
        if r[5] is not None: Z[i] = r[5]
    print(f"shell array {Z.shape} {Z.nbytes/1e9:.2f} GB", flush=True)

    clean = {}
    for i in np.where(obj == "none")[0]: clean[(body[i], view[i])] = i
    print(f"clean references: {len(clean)}", flush=True)
    cidx = np.array([clean[(b, v)] for b, v in zip(body, view)], np.int32)

    np.savez(OUT, Z=Z, y=y, body=body, view=view, obj=obj, name=name, clean_idx=cidx)
    print("wrote", OUT, f"{os.path.getsize(OUT)/1e9:.2f} GB", flush=True)
