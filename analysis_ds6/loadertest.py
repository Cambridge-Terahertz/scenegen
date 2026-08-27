#!/usr/bin/env python3
"""PHASE 5 ACCEPTANCE: does load_scan() turn a legacy file into the world frame correctly?

Ground truth is available because Phase 2 produced the SAME placement natively in both frames:
  ~/parity_legacy/... written by the old driver
  ~/parity_world/...  written by the migrated driver
So load_scan(legacy) must reproduce the native world file, not merely "look plausible".
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, "/home/jonah/probe")
import frame

L = {os.path.basename(f): f for f in glob.glob(os.path.expanduser("~/parity_legacy/**/*.npz"), recursive=True)}
W = {os.path.basename(f): f for f in glob.glob(os.path.expanduser("~/parity_world/**/*.npz"), recursive=True)}
ok = True
for name in sorted(set(L) & set(W)):
    conv = frame.load_scan(L[name])                    # legacy -> world via the loader
    nat  = frame.load_scan(W[name])                    # already world
    print(f"--- {name}")
    print(f"    converted frame tag: {conv['frame']}  (from {conv.get('frame_converted_from','native')})")
    for k in ("xs", "ys", "zs"):
        a, b = np.asarray(conv[k], float), np.asarray(nat[k], float)
        if a.shape != b.shape:
            print(f"    *** {k} shape {a.shape} vs {b.shape}"); ok = False; continue
        e = np.abs(a - b).max()
        print(f"    {k:4s} max err {e:.2e}  ({'OK' if e < 1e-5 else '*** MISMATCH'})"); ok &= e < 1e-5
    for k, tol in (("han_db", 0.15), ("depth_map", 1e-4)):
        a, b = np.asarray(conv[k], float), np.asarray(nat[k], float)
        if a.shape != b.shape:
            print(f"    *** {k} shape {a.shape} vs {b.shape}"); ok = False; continue
        m = np.isfinite(a) & np.isfinite(b)
        e = np.abs(a[m] - b[m]).max()
        print(f"    {k:10s} shape {a.shape}  max err {e:.4f}  ({'OK' if e < tol else '*** MISMATCH'})")
        ok &= e < tol
    for k in ("seat_xyz", "obj_aabb"):
        a, b = np.asarray(conv[k], float), np.asarray(nat[k], float)
        if not (np.isfinite(a).all() and np.isfinite(b).all()): continue
        e = np.abs(a - b).max()
        print(f"    {k:10s} max err {e:.5f} m  ({'OK' if e < 5e-3 else '*** MISMATCH'})"); ok &= e < 5e-3
    a, b = np.asarray(conv["obj_vox"]), np.asarray(nat["obj_vox"])
    if a.size and b.size:
        sa = set(map(tuple, a)); sb = set(map(tuple, b))
        j = len(sa & sb) / max(len(sa | sb), 1)
        print(f"    obj_vox    {len(sa)} vs {len(sb)} voxels, Jaccard {j:.4f}  ({'OK' if j > 0.97 else '*** MISMATCH'})")
        ok &= j > 0.97
    a, b = np.asarray(conv["gate_cells"]), np.asarray(nat["gate_cells"])
    if a.size and b.size:
        j = len(set(a.tolist()) & set(b.tolist())) / max(len(set(a.tolist()) | set(b.tolist())), 1)
        print(f"    gate_cells {len(a)} vs {len(b)}, Jaccard {j:.4f}  ({'OK' if j > 0.99 else '*** MISMATCH'})")
        ok &= j > 0.99
print("\n" + "="*58)
print("PHASE 5 ACCEPTANCE:", "PASS" if ok else "FAIL")
print("="*58)
