import os, sys, glob, numpy as np
os.environ.setdefault("SMPLX_MODELS", os.path.expanduser("~/smplx_models"))
os.environ.setdefault("AMASS_ROOT", os.path.expanduser("~/amass"))
for p in (os.path.expanduser("~/walkthrough"), os.path.expanduser("~/Documents/code/model_gen/python")):
    if os.path.isdir(p): sys.path.insert(0, p)
sys.path.insert(0, "/home/jonah/probe")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
from stance import screening_pose
import body as bodylib, anatomy, attach, threats
from mesh_sample import sample_mesh

STANDOFF, ROLL, GAP = 2.0, 0.0, 0.012
SITE, TURN = "wb_back_mid", True
D = "/home/jonah/dataset4_2026-07-21"
d = np.load(D + "/ds4_m_h50_b50_back_pistol_wb_back_mid.npz", allow_pickle=True)
betas = np.array(d["betas"]); gender = str(d["gender"])
XS = np.array(d["xs"]); ZS = np.array(d["zs"]); stored = np.array(d["depth_map"], float)

model = bodylib.get_model(gender); VC = bodylib.canonical_verts(model)
Vp, F = bodylib.posed_verts(model, betas=betas, body_pose=screening_pose())
m = trimesh.Trimesh(vertices=Vp, faces=F, process=False); m.fix_normals()
zc = 0.5 * (Vp[:, 1].min() + Vp[:, 1].max())

def to_sim(P, turn, z_center, is_dir=False):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    if turn: x, z = -x, -z
    if is_dir: return np.column_stack([x, -z, y])
    return np.column_stack([x, STANDOFF - z, y - z_center])

bq, bn, _ = sample_mesh(np.asarray(m.vertices), np.asarray(m.faces), 0.0006, front_cull=False)
BQ = to_sim(bq.astype(np.float64), TURN, zc); BN = to_sim(bn.astype(np.float64), TURN, zc, is_dir=True)
print("body pts", BQ.shape)

STL = next(p for p in glob.glob("/home/jonah/probe/glock17_ext.stl") +
           glob.glob("/home/jonah/**/glock17_ext.stl", recursive=True) if os.path.exists(p))
PHONE = os.path.join(os.path.dirname(STL), "phone_hull.stl")
obj = threats.build_library(glock_stl=STL, phone_stl=PHONE)["pistol"]
pl = anatomy.sample_placement(model, SITE, np.random.default_rng(0), jitter=False)
pl.update(roll=ROLL, standoff=GAP, flip=False)
gq, gn, gs = attach.attach(obj, Vp, VC, F, pl, target_nn=0.0006, vert_normals=m.vertex_normals)
GQ = to_sim(gq, TURN, zc); GN = to_sim(gn, TURN, zc, is_dir=True)
print("gun pts", GQ.shape)

STEP = 0.002
xs = np.arange(XS[0], XS[-1]+1e-9, STEP); zs = np.arange(ZS[0], ZS[-1]+1e-9, STEP)
dm = np.full((len(xs), len(zs)), np.inf); nn = np.zeros((len(xs), len(zs), 3)); tag = np.zeros((len(xs), len(zs)), np.uint8)
for pts, nrm, t in ((BQ, BN, 1), (GQ, GN, 2)):
    o = np.argsort(-pts[:, 1])                       # farthest first -> nearest overwrites
    ix = np.clip(np.round((pts[o,0]-xs[0])/STEP).astype(np.int32), 0, len(xs)-1)
    iz = np.clip(np.round((pts[o,2]-zs[0])/STEP).astype(np.int32), 0, len(zs)-1)
    cur = dm[ix, iz]; keep = pts[o,1] < cur
    ix, iz = ix[keep], iz[keep]
    dm[ix, iz] = pts[o,1][keep]; nn[ix, iz] = nrm[o][keep]; tag[ix, iz] = t
vis = np.isfinite(dm) & (tag > 0)
L = np.array([-0.45, -1.0, 0.55]); L /= np.linalg.norm(L)
inten = np.clip((nn @ L), 0, 1)**0.9 * 0.8 + 0.22
img = np.zeros(dm.shape + (3,))
img[tag == 1] = np.array([0.72, 0.78, 0.80]) * inten[tag == 1][:, None]
img[tag == 2] = np.array([0.95, 0.35, 0.20]) * inten[tag == 2][:, None]
img = np.clip(img, 0, 1); img[~vis] = 1.0

fig, ax = plt.subplots(1, 3, figsize=(13.5, 7.0))
sd = np.ma.masked_where(~np.isfinite(stored), -np.nan_to_num(stored, nan=0))
ax[0].imshow(sd.T, origin="lower", extent=[XS[0],XS[-1],ZS[0],ZS[-1]], cmap="bone",
             aspect="equal", interpolation="nearest")
ax[0].set_title("STORED in npz: depth_map\n1 cm raster, %d x %d" % stored.shape, fontsize=9)
ax[0].set_xlabel("<- what my earlier figures drew", fontsize=8)
for a in (ax[1], ax[2]):
    a.imshow(np.transpose(img, (1,0,2)), origin="lower",
             extent=[xs[0],xs[-1],zs[0],zs[-1]], aspect="equal", interpolation="bilinear")
ax[1].set_title("TRUE SMPL-X mesh + seated Glock 17\n2 mm render, %d verts / %d faces" % (Vp.shape[0], F.shape[0]), fontsize=9)
ax[2].set_title("zoom: gun on lower back", fontsize=9)
ax[2].set_xlim(-0.22, 0.22); ax[2].set_ylim(-0.05, 0.32)
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.suptitle("m_h50_b50 / back / pistol @ wb_back_mid - stored raster vs actual geometry", fontsize=11)
fig.savefig("/home/jonah/probe/ds4_meshtrue.png", dpi=150, bbox_inches="tight")
print("saved ds4_meshtrue.png")
