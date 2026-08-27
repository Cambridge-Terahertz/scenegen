#!/usr/bin/env python3
"""PHASE 3 ACCEPTANCE: the patched packers must extract the SAME physical quantities.

The conversion is a relabelling, so quantities that do not depend on the frame -- bbox extent,
voxel count, visibility -- must be unchanged. Quantities that DO depend on it -- the bbox centre x,
the image itself -- must be mirrored, not merely different.
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, "/home/jonah/probe")
import frame

D = "/mnt/shared_drive/Jonah/synthetic_data/dataset6_2026-08-25"
fs = sorted(glob.glob(os.path.join(D, "**", "*.npz"), recursive=True))[:200]
print(f"checking {len(fs)} pilot scans (all legacy-frame)\n")
bad = 0
for f in fs[:6]:
    raw = np.load(f, allow_pickle=True)
    w   = frame.load_scan(f)
    ov_r, ov_w = np.asarray(raw["obj_vox"]), np.asarray(w["obj_vox"])
    if not ov_r.size: continue
    lo_r, hi_r = ov_r.min(0), ov_r.max(0)
    lo_w, hi_w = ov_w.min(0), ov_w.max(0)
    # frame-INDEPENDENT: extent on the two cross-range axes and the voxel count
    wid_r, hei_r = hi_r[0]-lo_r[0]+1, hi_r[2]-lo_r[2]+1      # legacy image plane = (x, z)
    wid_w, hei_w = hi_w[0]-lo_w[0]+1, hi_w[1]-lo_w[1]+1      # world  image plane = (x, y)
    # frame-DEPENDENT: the centre x must mirror about the grid
    nx = len(raw["xs"])
    cx_r = (lo_r[0]+hi_r[0])/2
    cx_w = (lo_w[0]+hi_w[0])/2
    mirrored = abs(cx_w - ((nx-1) - cx_r)) < 1e-6
    same_shape = (wid_r == wid_w) and (hei_r == hei_w) and len(ov_r) == len(ov_w)
    ok = same_shape and mirrored
    bad += not ok
    print(f"{os.path.basename(f)[:52]:54s} extent {wid_r}x{hei_r} -> {wid_w}x{hei_w}  "
          f"nvox {len(ov_r)}->{len(ov_w)}  cx {cx_r:.1f}->{cx_w:.1f} (mirror {mirrored})  "
          f"{'OK' if ok else '*** MISMATCH'}")
# han_db must be the x-mirror
raw = np.load(fs[1], allow_pickle=True); w = frame.load_scan(fs[1])
e = np.abs(np.asarray(raw["han_db"])[::-1, :] - np.asarray(w["han_db"])).max()
print(f"\nhan_db legacy[::-1] vs world: max diff {e:.2e}  ({'OK - exact mirror' if e == 0 else '*** NOT a pure mirror'})")
bad += e != 0
print("\n" + "="*58)
print("PHASE 3 ACCEPTANCE:", "PASS" if bad == 0 else f"FAIL ({bad})")
print("="*58)
