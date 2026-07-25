"""Generate shapes_7x7.json = 98 bodies (2 gender x 7 height x 7 build) for dataset4.

Same SAMPLE -> MEASURE -> STRATIFY logic as stop_pose/shapes.py, with the identical pool
(seed 7, NPOOL=600 per gender) so population_estimate.py's NHANES comparison stays valid.
Only the number of strata changes. Runs on tesla (imports body from ~/walkthrough).

Height band names = band CENTRES of linspace(0,100,NLEV+1); build names = PCTS. At NLEV=5
this reproduces h10..h90 / b10..b90 exactly, so the convention is unchanged.
"""
import json, os, sys
import numpy as np
import trimesh

os.environ.setdefault("SMPLX_MODELS", os.path.expanduser("~/smplx_models"))
os.environ.setdefault("AMASS_ROOT", os.path.expanduser("~/amass"))
for p in (os.path.expanduser("~/walkthrough"),
          os.path.expanduser("~/Documents/code/model_gen/python")):
    if os.path.isdir(p):
        sys.path.insert(0, p)
import body as bodylib  # noqa: E402

NPOOL = 600
NLEV = int(os.environ.get("SHAPES_LEVELS", "7"))
OUT = os.path.expanduser(f"~/probe/shapes_{NLEV}x{NLEV}.json")
BODY_DENSITY = 1010.0
NHANES = {"male": (175.1, 29.4), "female": (161.2, 30.0)}

edges = np.linspace(0, 100, NLEV + 1)
HNAMES = [f"h{int(round((edges[i] + edges[i+1]) / 2)):02d}" for i in range(NLEV)]
PCTS = tuple(np.linspace(10, 90, NLEV))
BNAMES = [f"b{int(round(p)):02d}" for p in PCTS]
print(f"NLEV={NLEV} -> {2*NLEV*NLEV} bodies\n  heights {HNAMES}\n  builds  {BNAMES}\n")


def measure(model, betas):
    V, F = bodylib.posed_verts(model, betas=betas, body_pose=np.zeros(63))
    m = trimesh.Trimesh(vertices=V, faces=F, process=False)
    stature = float(V[:, 1].max() - V[:, 1].min())
    mass = abs(float(m.volume)) * BODY_DENSITY
    bmi = mass / stature ** 2
    y = V[:, 1].min() + 0.58 * stature
    sec = m.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
    waist = float(sec.to_2D()[0].length) * 100 if sec is not None else float("nan")
    return stature * 100, mass, bmi, waist


rng = np.random.default_rng(7)
chosen = []
for gender in ("male", "female"):
    model = bodylib.get_model(gender)
    B = rng.standard_normal((NPOOL, 10))
    rows = np.array([measure(model, b) for b in B])
    print(f"{gender}: stature {rows[:,0].mean():.1f}+-{rows[:,0].std():.1f} cm "
          f"(NHANES {NHANES[gender][0]}), BMI {rows[:,2].mean():.1f}+-{rows[:,2].std():.1f} "
          f"(NHANES {NHANES[gender][1]})", flush=True)
    h_edges = np.percentile(rows[:, 0], edges)
    for hi, hname in enumerate(HNAMES):
        band = np.flatnonzero((rows[:, 0] >= h_edges[hi]) & (rows[:, 0] <= h_edges[hi + 1]))
        bmis = rows[band, 2]
        for bname, p in zip(BNAMES, PCTS):
            j = band[np.argmin(np.abs(bmis - np.percentile(bmis, p)))]
            s, kg, bmi, w = rows[j]
            chosen.append(dict(gender=gender, height=hname, build=bname,
                               betas=[round(float(v), 4) for v in B[j]],
                               stature_cm=round(float(s), 1), mass_kg=round(float(kg), 1),
                               bmi=round(float(bmi), 1), waist_cm=round(float(w), 1)))

uniq = {(c["gender"], c["height"], c["build"]) for c in chosen}
dup_betas = len(chosen) - len({tuple(c["betas"]) for c in chosen})
print(f"\n{len(chosen)} bodies, {len(uniq)} unique (gender,height,build) cells, "
      f"{dup_betas} repeated beta vectors")
print(f"  stature {min(c['stature_cm'] for c in chosen):.1f}-"
      f"{max(c['stature_cm'] for c in chosen):.1f} cm   "
      f"BMI {min(c['bmi'] for c in chosen):.1f}-{max(c['bmi'] for c in chosen):.1f}")
with open(OUT, "w") as f:
    json.dump(chosen, f, indent=1)
print(f"-> {OUT}")
