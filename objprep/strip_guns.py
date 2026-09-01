"""Subdivide-then-strip. Face granularity limits how much interior can be removed: a face larger
than the rasteriser pixel is kept in FULL even when mostly occluded, so a coarse mesh keeps
interior it should lose. Refining to a sub-pixel edge first makes the result depend on the object,
not on how finely the artist happened to tessellate it."""
import os, sys, numpy as np, trimesh
sys.path.insert(0, os.path.expanduser("~/gunwork"))
import exterior_shell2 as ext_mod

V5  = os.path.expanduser("~/probe/objects_v5")
OUT = os.path.expanduser("~/gunwork/objects_v5_fixed"); os.makedirs(OUT, exist_ok=True)
MAX_EDGE = 0.0010                                  # 1 mm, below the 1.5 mm pixel
GUNS = ["gun1_glock17", "gun2_glock19", "gun3_hellcat", "gun4_mp9", "gun5_derringer"]
print("real pistol exterior is 350-450 cm2 (derringer much less)\n")
for n in GUNS:
    m = trimesh.load(os.path.join(V5, n + ".obj"), process=True)
    e0 = np.sqrt(m.area / len(m.faces)) * 1000
    n0 = len(m.faces)
    if e0 > MAX_EDGE * 1000:
        v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=MAX_EDGE)
        m = trimesh.Trimesh(v, f, process=True)
    e1 = np.sqrt(m.area / len(m.faces)) * 1000
    a0 = m.area
    e, st = ext_mod.strip_interior(m)
    e.fix_normals(); e.export(os.path.join(OUT, n + "_ext.obj"))
    print(f"{n:16s} {n0:9,d}->{len(m.faces):9,d} faces  edge {e0:5.2f}->{e1:4.2f}mm | "
          f"area {a0*1e4:6.1f} -> {st['area_out']*1e4:6.1f} cm2  (-{100*(1-st['area_out']/a0):4.1f}%)")
