#!/usr/bin/env python3
"""Relabel ds6 zero-visibility scans -- target list taken from the metadata table.

ds6_meta.py walked the SAME sorted glob and Pool.map preserves order with 0 errors, so row i of
ds6_meta.npz corresponds to file i. That lets us skip a second 31,920-file CIFS scan and open
only the ~1,559 files that actually need rewriting.

Rationale for the relabel: an object contributing no visible points leaves nothing in the image,
so a "threat" label there is pure noise. Original label kept as `is_threat_orig`.
"""
import numpy as np, glob, os, sys, tempfile
from multiprocessing import Pool
from collections import Counter

DATA = "/mnt/shared_drive/Jonah/synthetic_data/dataset6_2026-08-25"
DRY  = os.environ.get("DRY", "0") == "1"

def rewrite(f):
    try:
        with np.load(f, allow_pickle=True) as z:
            if "zero_vis" in z.files:
                return ("already", f)
            d = {k: z[k] for k in z.files}
        assert float(d["n_visible"]) == 0 and str(d["obj_name"]) != "none", "target mismatch"
        was = bool(d["is_threat"])
        d["zero_vis"]       = np.array(True)
        d["is_threat_orig"] = np.array(was)
        d["relabel_reason"] = np.array("zero_visibility")
        if was: d["is_threat"] = np.array(False)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(f), suffix=".tmp"); os.close(fd)
        np.savez_compressed(tmp, **d)
        os.replace(tmp + ".npz" if os.path.exists(tmp + ".npz") else tmp, f)
        with np.load(f, allow_pickle=True) as chk:
            assert bool(chk["zero_vis"]) and bool(chk["is_threat"]) is False
            assert chk["han_db"].shape == d["han_db"].shape
        return ("flipped" if was else "flagged", f)
    except Exception as e:
        return ("ERR", f"{os.path.basename(f)}: {e}")

if __name__ == "__main__":
    m = np.load(os.path.expanduser("~/probe/ds6_meta.npz"), allow_pickle=True)
    fs = np.array(sorted(glob.glob(os.path.join(DATA, "**", "*.npz"), recursive=True)))
    assert len(fs) == len(m["n_visible"]), f"glob {len(fs)} != meta {len(m[n_visible])}"
    mask = (m["obj_name"].astype(str) != "none") & (m["n_visible"].astype(float) == 0)
    y = m["is_threat"].astype(bool)
    targets = fs[mask]
    print(f"{len(fs)} files -> {mask.sum()} zero-visibility "
          f"({(mask & y).sum()} currently labelled THREAT, {(mask & ~y).sum()} benign)", flush=True)
    # sanity: the file name must contain the object and site the metadata says
    for f, o, s in list(zip(targets, m["obj_name"].astype(str)[mask], m["site"].astype(str)[mask]))[:3]:
        assert o in os.path.basename(f) and s in os.path.basename(f), f"ALIGNMENT FAIL {f} {o} {s}"
    print("  alignment spot-check OK", flush=True)
    if DRY: print("DRY RUN - nothing written"); sys.exit(0)
    with Pool(12) as p:
        out = p.map(rewrite, list(targets), chunksize=8)
    c = Counter(k for k, _ in out)
    print("  ", dict(sorted(c.items())), flush=True)
    for k, v in out:
        if k == "ERR": print("  ERR:", v)
    sys.exit(1 if c.get("ERR") else 0)
