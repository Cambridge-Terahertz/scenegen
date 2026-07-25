#!/usr/bin/env python
"""dataset4 detector analysis (plan sec.7 step 5).

Reports, on the SITE-MATCHED subset as primary (pooled only as a leak measurement):
  1. single-feature rank-AUC (threat vs benign) for each feature family
  2. per-confuser breakdown (which benign object is the worst false alarm)
  3. GroupKFold-over-body classifiers per feature GROUP, with a shuffle-label control
  4. per-confuser false-alarm rate at a fixed detection operating point

Feature families (see ds4_features.py):
  BRIGHT / SHAPE2D / RANGE3D  = BLIND (deployable);  PROBE_* = label-located;  CEILING = truth.
"""
import csv, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

FEAT = "/Users/jonah/.claude/jobs/dab08409/tmp/ds4_feat.csv"
R = list(csv.DictReader(open(FEAT)))
for r in R:
    r["is_threat"] = r["is_threat"] == "True"
obj = [r for r in R if r["obj_name"] != "none"]

GROUPS = {
    "BRIGHT":   ["hb_max", "hb_p90", "hb_p99", "hb_dyn"],
    "SHAPE2D":  [k for k in R[0] if k.startswith("bl")],
    "RANGE3D":  [k for k in R[0] if k.startswith("rng_")],
    "PROBE_shape": ["ow_area", "ow_asp", "ow_fill"],
    "PROBE_range": ["ow_rng_offset", "ow_rng_spread", "ow_rng_thick", "ow_rng_peakoff"],
    "CEILING":  ["n_obj_vox", "contrast_db"],
}
ALL_BLIND = GROUPS["BRIGHT"] + GROUPS["SHAPE2D"] + GROUPS["RANGE3D"]

def fget(rows, cols):
    return np.array([[float(r[c]) for c in cols] for r in rows], float)

def auc1(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    a = np.concatenate([pos, neg]); o = a.argsort()
    rk = np.empty(len(a)); rk[o] = np.arange(len(a))
    s = a[o]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            rk[o[i:j + 1]] = rk[o[i:j + 1]].mean()
        i = j + 1
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))

# ---- site-matched subset ----
st = {r["site"] for r in obj if r["is_threat"]}; sb = {r["site"] for r in obj if not r["is_threat"]}
shared = st & sb
SM = [r for r in obj if r["site"] in shared]
print(f"{len(obj)} object scans; site-matched {len(SM)} "
      f"({sum(r['is_threat'] for r in SM)} threat / {sum(not r['is_threat'] for r in SM)} benign)")
print(f"shared sites: {sorted(shared)}\n")

# ---- 1. single-feature rank-AUC, ALL vs SITE-MATCHED ----
print("=== single-feature rank-AUC (threat vs benign) ===")
print(f"{'feature':16s}{'family':12s}{'AUC_all':>9}{'AUC_sitematched':>17}")
key_feats = [("contrast_db","CEILING"),("n_obj_vox","CEILING"),("hb_max","BRIGHT"),("hb_p99","BRIGHT"),
             ("ow_area","PROBE_shape"),("ow_asp","PROBE_shape"),("ow_fill","PROBE_shape"),
             ("ow_rng_thick","PROBE_range"),("ow_rng_spread","PROBE_range"),("ow_rng_peakoff","PROBE_range"),
             ("bl10_maxarea","SHAPE2D"),("bl10_maxasp","SHAPE2D"),
             ("rng_area_off1cm","RANGE3D"),("rng_p99spread","RANGE3D"),("rng_blob_maxarea","RANGE3D")]
for f, fam in key_feats:
    if f not in R[0]: continue
    aa = auc1([float(r[f]) for r in obj if r["is_threat"]], [float(r[f]) for r in obj if not r["is_threat"]])
    asm = auc1([float(r[f]) for r in SM if r["is_threat"]], [float(r[f]) for r in SM if not r["is_threat"]])
    print(f"{f:16s}{fam:12s}{aa:9.3f}{asm:17.3f}")

# ---- 2. per-confuser: single-feature AUC of gun vs EACH benign, site-matched ----
print("\n=== gun vs each benign (site-matched rank-AUC; <0.5 => confuser beats gun on that feature) ===")
benigns = ["buckle", "phone", "name_badge", "wallet"]
show = ["contrast_db", "n_obj_vox", "ow_area", "ow_asp", "ow_rng_thick"]
print(f"{'benign':12s}" + "".join(f"{s:>14}" for s in show))
gpos = {s: [float(r[s]) for r in SM if r["is_threat"]] for s in show}
for b in benigns:
    row = [r for r in SM if r["obj_name"] == b]
    if not row: continue
    cells = "".join(f"{auc1(gpos[s], [float(r[s]) for r in row]):>14.3f}" for s in show)
    print(f"{b:12s}{cells}")

# ---- 3. GroupKFold classifiers per group (site-matched primary) ----
def cv_auc(rows, cols, shuffle=False, seed=0):
    X = fget(rows, cols); y = np.array([r["is_threat"] for r in rows], int)
    g = np.array([r["body"] for r in rows])
    if shuffle:
        rng = np.random.default_rng(seed); y = rng.permutation(y)
    oof = np.full(len(y), np.nan); aucs = []
    gk = GroupKFold(n_splits=5)
    for tr, te in gk.split(X, y, g):
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.06,
                                           l2_regularization=1.0, random_state=0)
        m.fit(X[tr], y[tr]); p = m.predict_proba(X[te])[:, 1]; oof[te] = p
        if len(set(y[te])) == 2: aucs.append(roc_auc_score(y[te], p))
    return roc_auc_score(y, oof), np.mean(aucs), np.std(aucs), oof, y

print("\n=== GroupKFold-over-body AUC, SITE-MATCHED (mean+-std across folds) ===")
print(f"{'feature group':16s}{'n_feat':>7}{'pooled':>8}{'fold mean':>11}{'std':>7}{'shuffle':>9}")
cvcache = {}
for name, cols in [("BRIGHT", GROUPS["BRIGHT"]), ("SHAPE2D", GROUPS["SHAPE2D"]),
                   ("RANGE3D", GROUPS["RANGE3D"]), ("ALL_BLIND", ALL_BLIND),
                   ("PROBE_shape", GROUPS["PROBE_shape"]), ("PROBE_range", GROUPS["PROBE_range"]),
                   ("PROBE_all", GROUPS["PROBE_shape"] + GROUPS["PROBE_range"]),
                   ("CEILING", GROUPS["CEILING"])]:
    cols = [c for c in cols if c in R[0]]
    pooled, fm, fs, oof, y = cv_auc(SM, cols)
    shuf, _, _, _, _ = cv_auc(SM, cols, shuffle=True)
    cvcache[name] = (oof, y)
    print(f"{name:16s}{len(cols):>7}{pooled:>8.3f}{fm:>11.3f}{fs:>7.3f}{shuf:>9.3f}")

# ---- 4. per-confuser false-alarm at fixed detection rate (best blind + best probe) ----
def fa_breakdown(rows, oof, y, pd_target=0.80):
    thr = np.quantile(oof[y == 1], 1 - pd_target)     # threshold catching pd_target of guns
    pd = (oof[y == 1] >= thr).mean()
    out = {"threshold_pd": pd}
    for b in benigns + ["ALL benign"]:
        if b == "ALL benign":
            mask = np.array([not r["is_threat"] for r in rows])
        else:
            mask = np.array([r["obj_name"] == b for r in rows])
        if mask.any():
            out[b] = float((oof[mask] >= thr).mean())
    return out

print("\n=== per-confuser false-alarm rate @ Pd=0.80 (site-matched, GroupKFold OOF) ===")
for name in ("ALL_BLIND", "PROBE_all", "CEILING"):
    oof, y = cvcache[name]
    fa = fa_breakdown(SM, oof, y)
    line = "  ".join(f"{k} {v:.2f}" for k, v in fa.items() if k != "threshold_pd")
    print(f"  {name:11s} (Pd={fa['threshold_pd']:.2f}):  {line}")

# ---- 5. pooled vs site-matched for the ceiling & best blind (the leak measurement) ----
print("\n=== pooled(all sites) vs site-matched, ALL_BLIND & CEILING (gap = location leak) ===")
for name, cols in [("ALL_BLIND", ALL_BLIND), ("CEILING", GROUPS["CEILING"])]:
    cols = [c for c in cols if c in R[0]]
    pa, _, _, _, _ = cv_auc(obj, cols); ps, _, _, _, _ = cv_auc(SM, cols)
    print(f"  {name:11s} pooled {pa:.3f}   site-matched {ps:.3f}   leak {pa-ps:+.3f}")
