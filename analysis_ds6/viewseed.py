#!/usr/bin/env python3
"""FIX 6 ACCEPTANCE: do the four views now share ONE physical placement?

Same body, object and site imaged from every view that sees it. The seated DOF -- bary (via
seat_xyz on the body), roll, gap, flip -- must now be IDENTICAL across views, where before they
were independently redrawn. The recon of course still differs: that is the whole point of having
four views.
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, "/home/jonah/probe")

OUT = os.path.expanduser("~/viewseed_test")
groups = {}
for f in sorted(glob.glob(os.path.join(OUT, "**", "*.npz"), recursive=True)):
    z = np.load(f, allow_pickle=True)
    if str(z["obj_name"]) == "none":
        continue
    key = (str(z["body"]), str(z["obj_name"]), str(z["site"]), int(z["placement_idx"]))
    groups.setdefault(key, []).append(
        (str(z["stance"]), float(z["roll_deg"]), float(z["gap_m"]) * 1000, bool(z["flip"]),
         int(z["face"]), tuple(np.round(np.asarray(z["bary"], float), 6))))

multi = {k: v for k, v in groups.items() if len(v) > 1}
print(f"{len(groups)} placements, {len(multi)} seen from more than one view\n")
ok = True
for k, rows in list(multi.items())[:6]:
    rolls = {round(r[1], 6) for r in rows}; gaps = {round(r[2], 6) for r in rows}
    flips = {r[3] for r in rows}; faces = {r[4] for r in rows}; barys = {r[5] for r in rows}
    same = len(rolls) == 1 and len(gaps) == 1 and len(flips) == 1 and len(faces) == 1 and len(barys) == 1
    ok &= same
    print(f"{k[1]} @ {k[2]}  views {[r[0] for r in rows]}")
    print(f"   roll {sorted(rolls)}  gap_mm {sorted(gaps)}  flip {flips}  face {faces}")
    print(f"   -> {'IDENTICAL across views  OK' if same else '*** STILL DIFFERS'}")
print("\n" + "="*58)
print("FIX 6 ACCEPTANCE:", "PASS" if ok and multi else "FAIL" if multi else "NO MULTI-VIEW PLACEMENTS")
print("="*58)
