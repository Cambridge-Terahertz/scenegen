#!/usr/bin/env python
"""Turn raw meshes into sim-ready assets: canonicalize (long=x, mid=y, thin=z), scale the
long axis to a real size, tag a material + a seat roll offset. Writes objects_v5/<name>.obj
(canonical, real-scale, metres) + manifest.json.  Sizes/materials are DEFAULTS — edit MANIFEST.
"""
import os, glob, json, numpy as np, sys
sys.path.insert(0, "/Users/jonah/Documents/scratch_results/threat_dataset")
import trimesh, threats
P = lambda a: float(np.max(a) - np.min(a))
BM = "/Users/jonah/Documents/Blender/Models"
TD = "/Users/jonah/Documents/scratch_results/threat_dataset"
OUT = os.path.join(TD, "objects_v5"); os.makedirs(OUT, exist_ok=True)
def g(p):
    m = sorted(glob.glob(p, recursive=True), key=len); return m[0] if m else None

# name: (source path, kind, material, target long-axis cm, seat roll deg)
MANIFEST = {
 # ---- confusers (nonthreat) ----
 "phone_nokia":   (g(f"{BM}/nonthreats/phone/*nokia*.obj"),        "benign","alloy",  12.5,180),
 "phone_modern":  (g(f"{BM}/nonthreats/phone/iphone16_slab.stl"),  "benign","alloy",  16.3,180),
 "wallet":        (g(f"{BM}/nonthreats/wallet/*Wallet*.obj"),      "benign","plastic",11.0,  0),
 "badge":         (g(f"{BM}/nonthreats/badge/variants/name_badge_1.obj"),"benign","polymer",9.0,180),
 "keys":          (g(f"{BM}/nonthreats/key/variants/keys_bunch.obj"),"benign","steel", 10.0,180),
 "necklace":      (g(f"{BM}/nonthreats/necklace/*.stl"),           "benign","steel",   5.0,  0),
 "belt":          (g(f"{BM}/nonthreats/belt/buckle/Buckle.obj"),   "benign","steel",   8.0, 90),
 # ---- threat guns ----
 "gun1_glock17":  (f"{TD}/glock17_ext.stl",                        "threat","steel",  19.5,  0),
 "gun2_glock19":  (g(f"{BM}/threats/gun/glock_cgtrader/**/Glock*.obj"),"threat","steel",18.7,0),
 "gun3_hellcat":  (g(f"{BM}/threats/gun/**/FMK 9C1.obj"),          "threat","steel",  16.0,  0),
 "gun4_mp9":      (g(f"{BM}/threats/gun/**/Grand Power K100.obj"), "threat","steel",  19.0,  0),
 "gun5_derringer":(g(f"{BM}/threats/gun/**/Tanfoglio GT27.obj"),   "threat","steel",  12.0,  0),
 # ---- threat knives ----
 "knife_switchblade":(g(f"{BM}/threats/knife/pocketknife_a/selected/switchblade.obj"),"threat","steel",11.0,0),
 "knife_folding":    (g(f"{BM}/threats/knife/pocketknife_a/selected/folding_pocket_knife.obj"),"threat","steel",11.0,0),
 "knife_corkscrew":  (g(f"{BM}/threats/knife/pocketknife_a/selected/corkscrew.obj"),"threat","steel",9.0,0),
 "knife_hunting":    (g(f"{BM}/threats/knife/pocketknife_a/selected/hunting_knife.obj"),"threat","steel",27.0,0),
 "knife_dielectric": (g(f"{BM}/threats/knife/pocketknife_a/selected/folding_pocket_knife.obj"),"threat","ceramic",11.0,0),
}

man = []
for name,(path,kind,mat,cm,roll) in MANIFEST.items():
    if not path or not os.path.exists(path):
        print(f"  MISSING {name}: {path}"); continue
    m = trimesh.load(path, process=True, force="mesh")
    if isinstance(m, trimesh.Scene): m = trimesh.util.concatenate(tuple(m.geometry.values()))
    V = threats._canon(np.asarray(m.vertices, np.float64))
    V *= (cm/100.0) / (V[:,0].max()-V[:,0].min())            # scale long axis to cm (metres)
    mm = trimesh.Trimesh(vertices=V, faces=np.asarray(m.faces), process=False)
    op = os.path.join(OUT, name+".obj"); mm.export(op)
    d = (V.max(0)-V.min(0))*100
    s = threats.MATERIALS[mat][0]
    man.append(dict(name=name, kind=kind, material=mat, s=s, roll_deg=roll,
                    dims_cm=[round(float(d[0]),1),round(float(d[1]),1),round(float(d[2]),2)],
                    verts=len(V), source=os.path.relpath(path, BM) if path.startswith(BM) else path))
    print(f"  {name:20s} {kind:6s} {mat:8s} s={s:.2f} {d[0]:5.1f}x{d[1]:4.1f}x{d[2]:4.1f}cm roll{roll:>4} v={len(V)}")

json.dump(man, open(os.path.join(OUT,"manifest.json"),"w"), indent=2)
print(f"\n{len(man)} objects -> {OUT}/  (+ manifest.json)")
