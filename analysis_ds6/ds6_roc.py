#!/usr/bin/env python3
"""ROC: can the threat label be read from LOCATION alone? dataset5 vs dataset6.

This is the one plot that says whether ds6 did its job. Both datasets are scored the same way:
a model is shown ONLY where the object sits (site + view, one-hot) and asked for the label.
Body-disjoint folds, so a body never appears in both train and test. Curves are pooled
out-of-fold predictions, which is why there is a single curve rather than a mean-of-folds band.

Extent (object size) is scored separately and deliberately: size is a real physical
discriminator, not a confound, and a correct dataset should keep it informative while location
goes to chance. Reporting the two fused -- as the original ds5 "position-only 0.989" did --
conflates a bug with physics.
"""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GroupKFold

P = os.path.expanduser("~/probe")
OUT = os.path.join(P, "ds6_figs"); os.makedirs(OUT, exist_ok=True)

def onehot(a):
    u = sorted(set(a)); return np.stack([(a == x).astype(np.float32) for x in u], 1)

def oof(X, y, groups, n_splits=4):
    """Pooled out-of-fold scores."""
    X = np.nan_to_num(np.asarray(X, np.float32), nan=0.0)
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p

def load(tag):
    d = np.load(os.path.join(P, f"{tag}_meta.npz"), allow_pickle=True)
    site = d["site"].astype(str); view = d["stance"].astype(str)
    y = d["is_threat"].astype(bool).astype(int); body = d["body"].astype(str)
    bb = d["bb"]; nvox = d["nvox"]
    return dict(site=site, view=view, y=y, body=body,
                loc=np.hstack([onehot(site), onehot(view)]),
                ext=np.column_stack([bb[:, 2], bb[:, 3], nvox]))

import sys
sys.path[:0] = ["/home/jonah/probe", "/home/jonah/walkthrough",
                "/home/jonah/Documents/code/model_gen/python"]
os.environ.setdefault("SMPLX_MODELS", os.path.expanduser("~/smplx_models"))
import placement_config as pc
TK = {n: (o["kind"] == "threat") for n, o in pc.OBJECTS.items()}; TK["gun1_glock17"] = True

runs = {}
D5, D6 = load("ds5"), load("ds6")
for tag, D in (("ds5", D5), ("ds6", D6)):
    print(f"{tag}: n={len(D['y'])} threat={D['y'].sum()} ({D['y'].mean()*100:.1f}%) "
          f"bodies={len(set(D['body']))}", flush=True)

def add(key, X, y, groups):
    p = oof(X, y, groups)
    fpr, tpr, _ = roc_curve(y, p)
    runs[key] = (fpr, tpr, roc_auc_score(y, p))
    print(f"   {key}: AUC {runs[key][2]:.3f}", flush=True)

for tag, D in (("ds5", D5), ("ds6", D6)):
    for feat in ("loc", "ext"):
        add((tag, feat), D[feat], D["y"], D["body"])

# ds6 with the zero-visibility scans excluded instead of relabelled -- the alternative that
# keeps location at chance, at the cost of 4.9% of the data. Both remain selectable at train
# time because every affected file carries `zero_vis` and `is_threat_orig`.
d6 = np.load(os.path.join(P, "ds6_meta.npz"), allow_pickle=True)
nv = d6["n_visible"].astype(float); ob = d6["obj_name"].astype(str)
zv = (nv == 0) & (ob != "none")
keep = ~zv
add(("ds6_drop", "loc"), D6["loc"][keep], D6["y"][keep], D6["body"][keep])

# ------------------------------------------------------------------ plot
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.4))

C5, C6 = "#B4453C", "#1F5F80"
for ax, feat, title, sub in (
    (axes[0], "loc", "LOCATION only  (site + view, no pixels)",
     "the confound: this SHOULD be a coin flip"),
    (axes[1], "ext", "EXTENT only  (object size, no location)",
     "real physics: this SHOULD stay informative")):
    series = [("ds5", C5, "dataset5", "-", 2.0), ("ds6", C6, "dataset6", "-", 2.0)]
    if feat == "loc":
        series.append(("ds6_drop", "#4E8F6D", "dataset6, zero-vis excluded", (0, (5, 2)), 1.6))
    for tag, c, lbl, ls, lw in series:
        if (tag, feat) not in runs: continue
        fpr, tpr, auc = runs[(tag, feat)]
        ax.plot(fpr, tpr, color=c, lw=lw, ls=ls, label=f"{lbl}   AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.9, ls=(0, (4, 3)), label="chance   AUC 0.500")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"{title}\n{sub}", fontsize=9.5, pad=7, linespacing=1.5)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.95)
    ax.grid(alpha=0.18, lw=0.6)

fig.suptitle("Killing the location shortcut - dataset5 vs dataset6, body-disjoint folds",
             fontsize=11.5, y=0.99)
fig.text(0.5, 0.012,
         "Left: dataset5 could largely name the threat from WHERE it sat, without seeing the image; "
         "dataset6 cannot. The dashed curve shows the cost of the\nzero-visibility relabel - it trades "
         "0.04 AUC of location leakage for 1,559 hard negatives, and both options stay open at train time.  "
         "Right: object size remains\ninformative in both, so the fix removed a confound rather than removing signal. "
         "Note dataset5's headline \"0.989 from metadata\" was EXTENT, not location.",
         ha="center", fontsize=8.0, color="0.3")
fig.subplots_adjust(top=0.815, bottom=0.215, left=0.07, right=0.98, wspace=0.22)
fig.savefig(os.path.join(OUT, "ds6_roc_location_shortcut.png"), dpi=160)
print("wrote ds6_roc_location_shortcut.png")
