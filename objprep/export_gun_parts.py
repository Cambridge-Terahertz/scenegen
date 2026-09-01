"""Export the final split assets at ORIGINAL mesh resolution.

Subdivision was a means, not an end: it existed so the rasteriser could resolve occlusion on
coarse meshes, and it added no geometric detail (it only splits flat triangles). Shipping it would
inflate gun2 from 612 KB to 54 MB for nothing. So the strip decision is mapped back onto the
original faces -- an original face is kept when at least half its area survived -- and the steel and
polymer parts are exported separately, which is also exactly the parts=[(mesh, material)] form
Threat already accepts. Splitting before export means the material boundary is exact rather than
being interpolated across a seam.
"""
import os, sys, numpy as np, trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.expanduser("~/gunwork"))
from gun_materials import GUNS, V5, FIX

OUT = os.path.expanduser("~/gunwork/objects_v6"); os.makedirs(OUT, exist_ok=True)
KEEP_FRAC = 0.5
MID_EDGE  = 0.0008     # ship at <=0.8 mm faces: fine enough that an all-or-nothing keep is
                       # accurate, coarse enough to stay a few MB. Well under the 5 mm range bin.
MAX_ROUNDS = 3         # uniform 4-way rounds, NOT subdivide_to_size: gun2 has ~20 mm slivers, and
                       # splitting until every EDGE fits drove it to 0.13 mm / 5 M faces / 54 MB.

print(f"{'mesh':16s} {'faces kept':>18s} {'area (cm2)':>22s}   {'steel%':>6s}")
summary = {}
for n, cfg in GUNS.items():
    src = trimesh.load(os.path.join(V5, n + ".obj"), process=True)
    ext = trimesh.load(os.path.join(FIX, n + "_ext.obj"), process=True)
    raw_area, raw_faces = src.area, len(src.faces)
    e0 = np.sqrt(src.area/len(src.faces))
    rounds = min(MAX_ROUNDS, int(np.ceil(np.log2(e0/MID_EDGE)))) if e0 > MID_EDGE else 0
    for _ in range(rounds):
        v, f = trimesh.remesh.subdivide(src.vertices, src.faces)
        src = trimesh.Trimesh(v, f, process=True)
    C0 = np.asarray(src.triangles_center); A0 = np.asarray(src.area_faces)
    Ce = np.asarray(ext.triangles_center); Ae = np.asarray(ext.area_faces)
    _, par = cKDTree(C0).query(Ce, k=1)                 # children lie inside their parent
    surv = np.zeros(len(A0)); np.add.at(surv, par, Ae)
    keep = surv >= KEEP_FRAC * A0

    lo, hi = np.asarray(src.vertices)[:,1].min(), np.asarray(src.vertices)[:,1].max()
    cut = hi - cfg["frac"]*(hi-lo) if cfg["slide_high"] else lo + cfg["frac"]*(hi-lo)
    if cfg["split"]:
        # split() neither preserves face order nor face count, so match on centroids
        pts, lab = [], []
        for c in trimesh.load(os.path.join(V5, n + ".obj"), process=True).split(only_watertight=False):
            cy = np.asarray(c.vertices)[:,1].mean()
            fc = np.asarray(c.triangles_center)
            pts.append(fc)
            lab.append(np.full(len(fc), (cy >= cut) if cfg["slide_high"] else (cy <= cut)))
        _, j = cKDTree(np.vstack(pts)).query(C0, k=1)
        steel = np.concatenate(lab)[j]
    else:
        steel = np.ones(len(A0), bool)

    parts = {}
    for tag, sel in (("steel", keep & steel), ("polymer", keep & ~steel)):
        if not sel.any():
            continue
        sub = src.submesh([np.where(sel)[0]], append=True)
        sub.fix_normals(); sub.export(os.path.join(OUT, f"{n}_{tag}.obj"))
        parts[tag] = sub.area*1e4
    a_s, a_p = parts.get("steel",0.0), parts.get("polymer",0.0)
    summary[n] = (a_s, a_p)
    print(f"{n:16s} {keep.sum():8,d}/{len(A0):8,d} {a_s+a_p:8.1f} "
          f"(vs {ext.area*1e4:6.1f} ref, {raw_area*1e4:6.1f} raw, {raw_faces:,} raw, +{rounds}r)"
          f"   {100*a_s/(a_s+a_p):5.1f}")
