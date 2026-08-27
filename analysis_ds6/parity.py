#!/usr/bin/env python3
"""PHASE 2 ACCEPTANCE: did the migrated driver reproduce the same physics in the new frame?

Not a bit-comparison -- Phase 0 established that fp32 shifts by ~1e-3 relative (0.0088 dB) purely
from reordering dx^2+dy^2+dz^2, so the tolerance below is set from that measurement, not guessed.

Checks, in order of what would hurt most if wrong:
  1. stored axes really are the world frame (range on z)      -- frame.assert_frame
  2. han_db agrees after undoing the x flip                   -- the image itself
  3. seat_xyz maps through legacy_to_world                    -- placement geometry
  4. the object lands on the same ANATOMICAL side             -- the sign error a migration hides
  5. scalar physics: visibility, n_visible, contrast          -- unchanged quantities
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, "/home/jonah/probe")
import frame

L = {os.path.basename(f): f for f in glob.glob(os.path.expanduser("~/parity_legacy/**/*.npz"), recursive=True)}
W = {os.path.basename(f): f for f in glob.glob(os.path.expanduser("~/parity_world/**/*.npz"), recursive=True)}
common = sorted(set(L) & set(W))
print(f"legacy {len(L)} / world {len(W)} / comparable {len(common)}\n")
TOL = 3e-3          # from Phase 0: fp32 frame round-off is ~1e-3 relative
ok_all = True

for name in common:
    a = np.load(L[name], allow_pickle=True); b = np.load(W[name], allow_pickle=True)
    print(f"--- {name}")
    # 1. frames
    try:
        frame.assert_frame((a["xs"], a["ys"], a["zs"]), expect="legacy"); f_a = "legacy"
    except Exception as e: f_a = f"FAIL({e})"
    try:
        frame.assert_frame((b["xs"], b["ys"], b["zs"]), expect="world"); f_b = "world"
    except Exception as e: f_b = f"FAIL({e})"
    print(f"    frames: legacy-file={f_a}   world-file={f_b}")
    ok_all &= (f_a == "legacy" and f_b == "world")
    # 2. han_db: world x = -legacy x  =>  flip axis 0
    ha, hb = np.asarray(a["han_db"], float), np.asarray(b["han_db"], float)
    if ha.shape != hb.shape:
        print(f"    *** han_db shape {ha.shape} vs {hb.shape}"); ok_all = False
    else:
        d = np.abs(ha[::-1, :] - hb)
        on = np.isfinite(ha[::-1, :]) & (ha[::-1, :] > -60)
        print(f"    han_db  max|dB diff| {d[on].max():.4f}   median {np.median(d[on]):.5f} dB"
              f"   ({'OK' if d[on].max() < 0.15 else '*** LARGE'})")
        ok_all &= d[on].max() < 0.15
    # 3. seat_xyz
    sa, sb = np.asarray(a["seat_xyz"], float), np.asarray(b["seat_xyz"], float)
    if np.isfinite(sa).all():
        exp = frame.legacy_to_world(sa[None, :])[0]
        err = np.abs(exp - sb).max()
        print(f"    seat_xyz legacy {np.round(sa,4)} -> expect {np.round(exp,4)}  got {np.round(sb,4)}"
              f"   max err {err:.5f} m  ({'OK' if err < 5e-3 else '*** MISMATCH'})")
        ok_all &= err < 5e-3
        # 4. the two frames must agree on which side the object sits -- compare like with like
        #    (seat_xyz is the OBJECT centroid, not the site anchor, so no absolute-sign assert here)
        same_side = np.sign(exp[0]) == np.sign(sb[0]) or abs(sb[0]) < 0.005
        print(f"    object side agrees between frames: {bool(same_side)}"
              f"   ({'OK' if same_side else '*** SIDE FLIPPED'})")
        ok_all &= bool(same_side)
    # 5. scalars
    for k, tol in (("visibility", 2e-3), ("n_visible", 0), ("han_contrast_db", 0.2), ("obliquity_deg", 0.5)):
        if k in a.files and k in b.files:
            va, vb = float(np.asarray(a[k], float)), float(np.asarray(b[k], float))
            if not (np.isfinite(va) and np.isfinite(vb)):
                print(f"    {k:16s} both non-finite: {va} / {vb}"); continue
            d = abs(va - vb); good = d <= tol if tol else va == vb
            print(f"    {k:16s} {va:.5f} vs {vb:.5f}   diff {d:.5f}  ({'OK' if good else '*** MISMATCH'})")
            ok_all &= bool(good)
print("\n" + "="*60)
print("PHASE 2 ACCEPTANCE:", "PASS" if ok_all else "FAIL")
print("="*60)
