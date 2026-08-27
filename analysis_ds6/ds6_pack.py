#!/usr/bin/env python3
"""Pack ds6 images for the background-model experiment.

Question: how much of the achievable image performance depends on having a CLEAN scan of the
same body in the same pose -- something that does not exist on real data, where you never image
the person without the threat?

So we store two views of every scan:
  raw  = han_db, what a real system would actually have
  diff = han_db - han_db(clean scan of the SAME body and view), an oracle background subtraction
Training on both and comparing separates "recognises the object" from "spots a deviation from a
memorised body template". Images are block-averaged 2x to 90x105 to keep this a fast experiment.
"""
import numpy as np, glob, os, sys
import sys as _sys; _sys.path.insert(0, "/home/jonah/probe")
import frame  # world-frame loader: normalises legacy files on read

from multiprocessing import Pool

DATA = "/mnt/shared_drive/Jonah/synthetic_data/dataset6_2026-08-25"
OUT  = os.path.expanduser("~/probe/ds6_img.npz")
NR, NC = 180, 210          # crop from 181x211
H, W = NR//2, NC//2        # 90 x 105

def down(a):
    a = np.asarray(a, np.float32)[:NR, :NC]
    return a.reshape(H, 2, W, 2).mean(axis=(1, 3))

def proc(f):
    try:
        z = frame.load_scan(f)          # world frame regardless of vintage
        han = down(z["han_db"])
        return (os.path.basename(f), str(z["body"]), str(z["stance"]),
                str(z["obj_name"]), bool(z["is_threat"]), han)
    except Exception as e:
        return ("ERR", str(e), None, None, None, None)

if __name__ == "__main__":
    fs = sorted(glob.glob(os.path.join(DATA, "**", "*.npz"), recursive=True))
    print(f"{len(fs)} files -> {H}x{W}", flush=True)
    with Pool(24) as p:
        rows = p.map(proc, fs, chunksize=48)
    bad = [r for r in rows if r[0] == "ERR"]
    print(f"errors: {len(bad)}", flush=True)
    rows = [r for r in rows if r[0] != "ERR"]

    name = np.array([r[0] for r in rows]); body = np.array([r[1] for r in rows])
    view = np.array([r[2] for r in rows]); obj = np.array([r[3] for r in rows])
    y    = np.array([r[4] for r in rows], bool)
    han  = np.stack([r[5] for r in rows]).astype(np.float32)

    # clean reference per (body, view) -- the baseline scan carrying no object
    clean = {}
    for i in np.where(obj == "none")[0]:
        clean[(body[i], view[i])] = han[i]
    print(f"clean references: {len(clean)} (expect bodies x views)", flush=True)
    miss = [k for k in zip(body, view) if k not in clean]
    if miss: print("MISSING clean refs:", len(set(miss)), flush=True)

    ref = np.stack([clean[(b, v)] for b, v in zip(body, view)])
    diff = han - ref

    # quantise: raw over [-60,0] dB, diff over [-15,15] dB
    Xr = np.clip((han + 60.0) / 60.0, 0, 1)
    Xd = np.clip((diff + 15.0) / 30.0, 0, 1)
    np.savez(OUT, Xr=(Xr*255).astype(np.uint8), Xd=(Xd*255).astype(np.uint8),
             y=y, body=body, view=view, obj=obj, name=name)
    print("wrote", OUT, Xr.shape, f"{os.path.getsize(OUT)/1e9:.2f} GB", flush=True)
