#!/usr/bin/env python3
"""ds6 pilot PRIMARY GATE: can the threat label be read from LOCATION alone?

ds5 scored 0.989 AUC from bbox metadata with no pixels. That number conflated two things:
  * WHERE the object sits  -- a pure confound, must go to 0.5
  * HOW BIG it is          -- a legitimate physical discriminator, must NOT go to 0.5
so they are scored separately here. Success = location ~0.50 while extent stays informative.
"""
import numpy as np, os, json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

d = np.load(os.path.expanduser("~/probe/ds6_meta.npz"), allow_pickle=True)
site = d["site"].astype(str); view = d["stance"].astype(str)
obj  = d["obj_name"].astype(str); y = d["is_threat"].astype(bool).astype(int)
body = d["body"].astype(str); bb = d["bb"]; nvox = d["nvox"]
vis  = d["visibility"].astype(float) if "visibility" in d.files else np.full(len(y), np.nan)

print(f"n={len(y)}  threat={y.sum()} ({y.mean()*100:.1f}%)  bodies={len(set(body))}  views={sorted(set(view))}")
print(f"objects={len(set(obj))}  sites={len(set(site))}")

# ---- 1. the exact confound check, measured on the DATA (not the design) ----
print("\n== marginal check: max |P(site|threat) - P(site|benign)| per view")
worst = 0.0
for v in sorted(set(view)):
    m = view == v
    st = sorted(set(site[m]))
    pt = np.array([(site[m & (y==1)] == s).mean() for s in st])
    pb = np.array([(site[m & (y==0)] == s).mean() for s in st])
    dmax = np.abs(pt-pb).max(); worst = max(worst, dmax)
    print(f"   {v:>5}: {len(st):>3} sites   max delta = {dmax:.3e}")
print(f"   WORST ACROSS VIEWS = {worst:.3e}")

# ---- 2. AUCs, body-disjoint ----
def onehot(a):
    u = sorted(set(a)); return np.stack([(a==x).astype(np.float32) for x in u], 1)

FEATS = {
  "location only (site+view)":      np.hstack([onehot(site), onehot(view)]),
  "extent only (w,h,nvox)":         np.column_stack([bb[:,2], bb[:,3], nvox]),
  "location + extent (ds5-style)":  np.hstack([onehot(site), onehot(view),
                                               np.column_stack([bb[:,2],bb[:,3],nvox])]),
  "centroid only (cx,cy)":          bb[:, :2],
}
gkf = GroupKFold(n_splits=4)
print("\n== ROC-AUC, body-disjoint 4-fold (chance = 0.500)")
res = {}
for name, X in FEATS.items():
    Xc = np.nan_to_num(X, nan=0.0)
    aucs = []
    for tr, te in gkf.split(Xc, y, groups=body):
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        clf.fit(Xc[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xc[te])[:,1]))
    m, s = float(np.mean(aucs)), float(np.std(aucs))
    res[name] = m
    print(f"   {name:<32} {m:.3f} +/- {s:.3f}")

print("\n== gate")
loc = res["location only (site+view)"]
print(f"   location-only AUC {loc:.3f}  (ds5 position-only was 0.989)")
print("   PASS - location carries no label information" if abs(loc-0.5) < 0.02
      else "   FAIL - location still predicts the label")
if not np.isnan(vis).all():
    zv = vis <= 0
    print(f"\n== zero-visibility scans: {zv.sum()} ({zv.mean()*100:.2f}%)  of which labelled threat: {(zv & (y==1)).sum()}")
    for v in sorted(set(view)):
        m = view == v
        print(f"   {v:>5}: {(zv&m).sum():>5} / {m.sum():>5}  ({(zv&m).sum()/m.sum()*100:.2f}%)")
json.dump(res, open(os.path.expanduser("~/probe/ds6_gate.json"),"w"), indent=1)
