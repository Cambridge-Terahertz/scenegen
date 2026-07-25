#!/usr/bin/env python
"""Render candidate THREAT meshes (guns + knives) object-only, 2 views each, so the
reviewer can judge the picks vs the spreadsheet. Shapes normalized to a common long-axis
(scale is set later); material color: steel=red, dielectric/ceramic=blue."""
import os, sys, glob, numpy as np
sys.path.insert(0, "/Users/jonah/Documents/scratch_results/threat_dataset")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import trimesh, threats
P = lambda a: float(np.max(a) - np.min(a))
GUNDIR = "/Users/jonah/Documents/Blender/Models/threats/gun"
KNIFEDIR = "/Users/jonah/Documents/Blender/Models/threats/knife"
OUT = "/Users/jonah/.claude/jobs/dab08409/tmp/gallery_imgs"

def find(pat):
    m = sorted(glob.glob(pat, recursive=True), key=len)
    return m[0] if m else None

# (label, path, material)  material: steel=metal(red), ceramic=dielectric(blue)
GUNS = [
    ("gun1 · Glock (dataset4)\n(glock17_ext · consistency)", "/Users/jonah/Documents/scratch_results/threat_dataset/glock17_ext.stl", "steel"),
    ("gun2 · Glock Gen5\n(Glock 19)",        find(f"{GUNDIR}/glock_cgtrader/**/Glock*.obj"), "steel"),
    ("gun3 · Hellcat ~=\n(FMK 9C1)",         find(f"{GUNDIR}/**/FMK 9C1.obj"), "steel"),
    ("gun4 · S&W M&P9 ~=\n(Grand Power K100)", find(f"{GUNDIR}/**/Grand Power K100.obj"), "steel"),
    ("gun5 · derringer ~=\n(Tanfoglio GT27)",find(f"{GUNDIR}/**/Tanfoglio GT27.obj"), "steel"),
]
KNIVES = [
    ("knife1 · pocket\n(military_pocket)",   find(f"{KNIFEDIR}/military_pocket_knife/*.obj"), "steel"),
    ("knife2 · 6\" fixed\n(hunting blade)",  find(f"{KNIFEDIR}/hunting_fixed_blade_army/*.obj"), "steel"),
    ("knife3 · dielectric\n(tactical+ceramic)", find(f"{KNIFEDIR}/military_tactical_knife/*.obj"), "ceramic"),
    ("knife4 · carving ~=\n(no exact match)", find(f"{KNIFEDIR}/military_tactical_knife/*.obj"), "steel"),
    ("knife5 · filler\n(karambit)",          find(f"{KNIFEDIR}/military_karambit_knife/*.obj"), "steel"),
]
COL = {"steel": "#d94a3d", "ceramic": "#4a4a8a"}

def render_group(items, title, fname):
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.2))
    for col, (lab, path, mat) in enumerate(items):
        if not path:
            axes[0, col].set_title(lab + "\n[MISSING]", fontsize=8);
            for r in (0,1): axes[r,col].set_xticks([]); axes[r,col].set_yticks([]); continue
        m = trimesh.load(path, process=True, force="mesh")
        if isinstance(m, trimesh.Scene): m = trimesh.util.concatenate(tuple(m.geometry.values()))
        V = threats._canon(np.asarray(m.vertices, np.float64))
        V = V / (V[:, 0].max() - V[:, 0].min())               # normalize long-axis -> 1
        if len(V) > 50000:
            V = V[np.random.default_rng(0).choice(len(V), 50000, replace=False)]
        c = COL[mat]
        axes[0, col].scatter(V[:, 0], V[:, 1], s=0.5, c=c, lw=0)   # profile (side)
        axes[0, col].set_aspect("equal"); axes[0, col].set_title(lab, fontsize=8.5)
        axes[1, col].scatter(V[:, 0], V[:, 2], s=0.5, c=c, lw=0)   # top (thin axis)
        axes[1, col].set_aspect("equal")
        axes[1, col].set_title(f"top · aspect T/L={P(V[:,2]):.2f}", fontsize=7.5)
        for r in (0, 1): axes[r, col].set_xticks([]); axes[r, col].set_yticks([])
    fig.suptitle(title + "   (shapes normalized to equal length; red=metal, blue=dielectric; "
                 "profile top / plan bottom)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=115, bbox_inches="tight")
    print("wrote", fname)

for lab, p, _ in GUNS + KNIVES:
    print(f"  {'OK ' if p else 'MISS'} {lab.splitlines()[0]:26s} {os.path.basename(p) if p else '---'}")
render_group(GUNS, "Threat guns — spreadsheet picks", "_threats_guns.png")
render_group(KNIVES, "Threat knives — spreadsheet picks", "_threats_knives.png")
