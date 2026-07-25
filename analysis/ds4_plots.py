#!/usr/bin/env python
"""dataset4 detection RESULTS figure (companion to ds4_ANALYSIS.md)."""
import csv, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, roc_curve

FEAT = os.environ.get("FEAT_CSV", "/Users/jonah/.claude/jobs/dab08409/tmp/ds4_feat.csv")
OUT = os.environ.get("PLOT_OUT", "/Users/jonah/.claude/jobs/dab08409/tmp/figpreview/fig7_detect_results.png")
R = [r for r in csv.DictReader(open(FEAT)) if r["obj_name"] != "none"]
for r in R: r["is_threat"] = r["is_threat"] == "True"
st = {r["site"] for r in R if r["is_threat"]}; sb = {r["site"] for r in R if not r["is_threat"]}
shared = st & sb; SM = [r for r in R if r["site"] in shared]

GROUPS = {"BRIGHT": ["hb_max","hb_p90","hb_p99","hb_dyn"],
          "SHAPE2D": [k for k in R[0] if k.startswith("bl")],
          "RANGE3D": [k for k in R[0] if k.startswith("rng_")],
          "PROBE_shape": ["ow_area","ow_asp","ow_fill"],
          "PROBE_range": ["ow_rng_offset","ow_rng_spread","ow_rng_thick","ow_rng_peakoff"],
          "CEILING": ["n_obj_vox","contrast_db"]}
GROUPS["ALL_BLIND"] = GROUPS["BRIGHT"] + GROUPS["SHAPE2D"] + GROUPS["RANGE3D"]

def fget(rows, cols): return np.array([[float(r[c]) for c in cols] for r in rows], float)
def cv(rows, cols, shuffle=False):
    X = fget(rows, cols); y = np.array([r["is_threat"] for r in rows], int)
    g = np.array([r["body"] for r in rows]); oof = np.full(len(y), np.nan)
    if shuffle: y = np.random.default_rng(0).permutation(y)
    for tr, te in GroupKFold(5).split(X, y, g):
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.06,
                                           l2_regularization=1.0, random_state=0)
        m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    return oof, y

OBJS = ["pistol","buckle","phone","name_badge","wallet"]
OCOL = {"pistol":"#d94a3d","buckle":"#7a7ac0","phone":"#7a7ac0","name_badge":"#9aa7b3","wallet":"#9aa7b3"}

fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.6))

# --- (1) HONEST decomposition staircase: what each tier buys, gap-cue flagged fragile ---
BR = ["hb_max","hb_p90","hb_p99","hb_dyn"]; SH = [k for k in R[0] if k.startswith("bl")]
THK = ["rng_maxspread","rng_p99spread","rng_area_spr1cm"]
GAP = ["rng_maxoffset","rng_p99offset","rng_proud_max","rng_area_off1cm","rng_blob_n","rng_blob_maxarea","rng_blob_maxasp"]
steps = [("brightness only", BR, "#9aa7b3"),
         ("+ 2D image shape", BR+SH, "#9aa7b3"),
         ("+ range thickness\n= transfer-honest blind", BR+SH+THK, "#2c7fb8"),
         ("+ gap cue (depth-anchored,\nFRAGILE ≈ leaky depth bump)", BR+SH+THK+GAP, "#d94a3d"),
         ("truth ceiling (n_obj_vox)", GROUPS["CEILING"], "#238b45")]
vals = [roc_auc_score(*(lambda o,y:(y,o))(*cv(SM, cols))) for _, cols, _ in steps]
yy = np.arange(len(steps))
ax[0,0].barh(yy, vals, 0.62, color=[c for *_, c in steps])
ax[0,0].axvline(0.5, color="k", ls="--", lw=1, label="chance / shuffle = 0.52")
for i, v in enumerate(vals): ax[0,0].text(v+.008, i, f"{v:.2f}", va="center", fontsize=9, fontweight="bold")
ax[0,0].set_yticks(yy); ax[0,0].set_yticklabels([s for s,_,_ in steps], fontsize=8)
ax[0,0].set_xlim(0.4,1.06); ax[0,0].invert_yaxis(); ax[0,0].set_xlabel("GroupKFold AUC (over 98 bodies), site-matched")
ax[0,0].set_title("A. What each feature tier buys — the 0.96 is almost\nall the FRAGILE gap cue; honest blind ≈ 0.73", fontsize=10)
ax[0,0].legend(fontsize=8, loc="lower right")

# --- (2) ROC curves, site-matched ---
for gname, c in [("BRIGHT","#7a7ac0"),("PROBE_shape","#e08214"),("ALL_BLIND","#2c7fb8"),("CEILING","#238b45")]:
    oof, yy = cv(SM, GROUPS[gname]); fpr, tpr, _ = roc_curve(yy, oof)
    ax[0,1].plot(fpr, tpr, c=c, lw=2, label=f"{gname}  AUC {roc_auc_score(yy,oof):.3f}")
ax[0,1].plot([0,1],[0,1],"k--",lw=1)
ax[0,1].set_xlabel("false-alarm rate (benign flagged)"); ax[0,1].set_ylabel("detection rate (guns caught)")
ax[0,1].set_title("B. ROC, site-matched (blind vs truth-ceiling)", fontsize=10); ax[0,1].legend(fontsize=8, loc="lower right")

# --- (3) per-confuser false alarm @ Pd=0.80 ---
def fa(oof, y, rows, pd=0.80):
    thr = np.quantile(oof[y==1], 1-pd)
    return {b: float((oof[np.array([r["obj_name"]==b for r in rows])] >= thr).mean()) for b in OBJS[1:]}
faB = fa(*cv(SM, GROUPS["ALL_BLIND"]), SM); faP = fa(*cv(SM, GROUPS["PROBE_shape"]+GROUPS["PROBE_range"]), SM)
xb = np.arange(len(OBJS)-1); w=0.38
ax[0,2].bar(xb-w/2, [faB[b] for b in OBJS[1:]], w, color="#2c7fb8", label="ALL_BLIND")
ax[0,2].bar(xb+w/2, [faP[b] for b in OBJS[1:]], w, color="#e08214", label="PROBE (located)")
ax[0,2].set_xticks(xb); ax[0,2].set_xticklabels(OBJS[1:], fontsize=9)
ax[0,2].set_ylabel("false-alarm rate @ Pd=0.80"); ax[0,2].legend(fontsize=8)
ax[0,2].set_title("C. Worst confusers = the two metals (buckle, phone)", fontsize=10)

# --- box helper ---
def box(a, feat, title, ylab, logy=False):
    data = [np.array([float(r[feat]) for r in SM if r["obj_name"]==o]) for o in OBJS]
    bp = a.boxplot(data, labels=OBJS, patch_artist=True, showfliers=False, widths=0.6)
    for patch,o in zip(bp["boxes"], OBJS): patch.set_facecolor(OCOL[o]); patch.set_alpha(.85)
    for med in bp["medians"]: med.set_color("k")
    a.axhline(np.median(data[0]), color="#d94a3d", ls=":", lw=1.2)  # gun median ref
    if logy: a.set_yscale("log")
    a.set_ylabel(ylab); a.set_title(title, fontsize=10); a.tick_params(axis="x", labelsize=9)

# --- (4) contrast by object, (5) range footprint, (6) image area (inversion) ---
box(ax[1,0], "contrast_db", "D. Brightness (contrast): buckle & phone ≥ gun\n(dotted = gun median)", "contrast dB")
box(ax[1,1], "rng_blob_maxarea", "E. Range-3D 'off-surface footprint' — gun biggest,\nBUT it's the air-gap cue (≈ leaky depth bump)", "off-surface blob area [cells]")
box(ax[1,2], "ow_area", "F. Image shape INVERTS: gun images compact,\nflat slabs are bigger bright blobs", "bright-blob area @ object [px]")

fig.suptitle("dataset4 — gun vs benign detection (site-matched: 882 threat / 1,862 benign, GroupKFold over 98 bodies)\n"
             "brightness overlaps · image-2D shape inverts · transfer-honest blind ≈ 0.73 · the 0.96 is the fragile depth-gap cue · truth-size ceiling = 1.00",
             fontsize=11)
fig.tight_layout(rect=(0,0,1,0.95))
fig.savefig(OUT, dpi=140, bbox_inches="tight"); print("saved", OUT)
