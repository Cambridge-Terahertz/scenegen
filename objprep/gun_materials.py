"""Steel/polymer assignment for the objects_v5 pistols.

WHY: all five guns are currently material=steel, s=1.0. Real polymer-framed pistols are a bright
steel slide over a much dimmer frame; modelling them as solid steel is both too bright overall and
erases the internal contrast structure a detector should be keying on.

TWO FIXES, IN ORDER -- interiors first, then materials:
  1. Interior stripping. These are artist solids carrying bore, magwell and frame internals that
     THz can never see. Measured: gun2 60% of its surface, gun3 65%, gun4 57%, gun5 61%. Splitting
     first computes fractions against surface that should not exist (gun2 first read 55% steel).
  2. Material split, per CONNECTED COMPONENT. Every mesh is built from separate solids (28-2791),
     the same structure the hand-verified Glock split used. Component assignment keeps the trigger
     guard and magazine intact instead of slicing them mid-face.

ORIENTATION IS DECLARED, NOT INFERRED. Two geometric detectors were tried and both failed: the
x-extent profile inverted gun1 (its angled grip spans as much length as the slide), and the
muzzle-end centroid landed at 0.50/0.47 on gun3/gun4 (the dust cover sits under the barrel, so the
muzzle slice straddles mid-height). With five one-time meshes, a field checked by eye against the
render is more trustworthy -- and auditable -- than a rule that can silently invert.

MATERIALS from the real weapons: Glock 17/19 and the FMK 9C1 and Grand Power K100 are all
polymer-framed with steel slides. The Tanfoglio GT27 is cast zinc alloy throughout: NOT split.
"""
import os, sys, numpy as np, trimesh
from scipy.spatial import cKDTree

V5   = os.path.expanduser("~/probe/objects_v5")
FIX  = os.path.expanduser("~/gunwork/objects_v5_fixed")

# slide_high: verified by eye in gun_materials.png. split=False -> uniform metal.
GUNS = {
    "gun1_glock17":   dict(real="Glock 17",         split=True,  slide_high=True,  frac=0.42),
    "gun2_glock19":   dict(real="Glock 19",         split=True,  slide_high=True,  frac=0.42),
    "gun3_hellcat":   dict(real="FMK 9C1",          split=True,  slide_high=True,  frac=0.42),
    "gun4_mp9":       dict(real="Grand Power K100", split=True,  slide_high=True,  frac=0.42),
    "gun5_derringer": dict(real="Tanfoglio GT27",   split=False, slide_high=True,  frac=0.42),
}


def material_faces(name):
    """-> (exterior mesh, bool steel-mask over its faces, info)"""
    cfg = GUNS[name]
    src = trimesh.load(os.path.join(V5, name + ".obj"), process=True)
    ext = trimesh.load(os.path.join(FIX, name + "_ext.obj"), process=True)
    if not cfg["split"]:
        return ext, np.ones(len(ext.faces), bool), dict(ncomp=0, **cfg)

    V = np.asarray(src.vertices)
    lo, hi = V[:, 1].min(), V[:, 1].max(); H = hi - lo
    cut = (hi - cfg["frac"] * H) if cfg["slide_high"] else (lo + cfg["frac"] * H)
    comps = src.split(only_watertight=False)
    pts, lab = [], []
    for c in comps:
        cy = np.asarray(c.vertices)[:, 1].mean()
        steel = (cy >= cut) if cfg["slide_high"] else (cy <= cut)
        fc = np.asarray(c.triangles_center)
        pts.append(fc); lab.append(np.full(len(fc), steel))
    pts = np.vstack(pts); lab = np.concatenate(lab)
    # stripping only DELETES faces, so every survivor lies on a labelled original face
    _, idx = cKDTree(pts).query(np.asarray(ext.triangles_center), k=1)
    return ext, lab[idx], dict(ncomp=len(comps), **cfg)


if __name__ == "__main__":
    for n in GUNS:
        ext, steel, d = material_faces(n)
        fa = np.asarray(ext.area_faces)
        a_s, a_p = fa[steel].sum()*1e4, fa[~steel].sum()*1e4
        note = "" if d["split"] else "   (all metal, not split)"
        print(f"{n:16s} {d['ncomp']:5d} comps | steel {a_s:6.1f} cm2 "
              f"({100*a_s/(a_s+a_p):4.1f}%)  polymer {a_p:6.1f} cm2{note}")
