"""dataset3 detection analysis — plan §7 step 5 (sim-internal, SHAPE features).

Answers, in order:
  PART 1  ORACLE: with ground-truth gun_vox, are gun blobs separable from body glints by SHAPE?
                  (dataset2 probe said 0.968 at 1 m / Hann; does it survive 2 m / Taylor?)
  PART 2  BLIND per-image detection: body-grouped CV, several feature families + a per-blob
          classifier aggregated by max. Shuffle-label control MUST land at ~0.5.
  PART 3  STRATIFY: AUC / score by carry site, stance, visibility — separating "model failed"
          from "physically not visible".

Axis convention (plan §7 step 5 bug-fix): sim `han_db` is (XS, ZS) = (x, z), so the body's
left-right mirror is **axis 0**. `np.fliplr` would flip VERTICAL and silently measure carry height.
Asserted below.

    python ds3_detect.py            # reads ds3_cache.npz, writes ds3_detect.png + stdout report
"""
import os, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "ds3_cache.npz")
OUT = os.environ.get("OUT", HERE)

from skimage.measure import label, regionprops
from skimage.morphology import white_tophat, disk, binary_dilation
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

NX, NZ = 181, 211
TOPHAT_R = 8                 # cm; isolates structures smaller than ~16 cm (gun ~19x13 cm)
THRESHES = [4.0, 8.0, 12.0, 16.0]   # dB above local background, on the top-hat map
AREA_MAX = 1200              # cm^2 — bigger than this is body structure, not a carried object
AREA_MIN = 4                 # cm^2 — below one resolution cell
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- data -------------
def load():
    z = np.load(CACHE, allow_pickle=True)
    han = z["han_db"].astype(np.float32)
    n = len(han)
    gun = np.unpackbits(z["gun_mask"])[: n * NX * NZ].reshape(n, NX, NZ).astype(bool)
    meta = {k: z[k] for k in z.files if k not in ("han_db", "gun_mask", "gun_shape")}
    assert han.shape[1:] == (NX, NZ), han.shape
    return han, gun, meta


# ------------------------------------------------------------- blob layer ----------
_FOOT = disk(TOPHAT_R)
DESCR = ["area", "aspect", "extent", "solidity", "ecc", "compact", "peak_db", "mean_db", "elong_area"]


def tophat(img):
    """Local-contrast map: dB above the smoothly-varying body background."""
    return white_tophat(img, footprint=_FOOT)


def blob_table(img, th, thr):
    """Connected components of (top-hat > thr). Returns (list-of-descriptor-vectors, list-of-masks)."""
    lbl = label(th > thr)
    feats, masks = [], []
    for rp in regionprops(lbl, intensity_image=img):
        if rp.area < AREA_MIN or rp.area > AREA_MAX:
            continue
        maj = max(rp.axis_major_length, 1e-6)
        mnr = max(rp.axis_minor_length, 1e-6)
        per = max(rp.perimeter, 1e-6)
        feats.append([rp.area, maj / mnr, rp.extent, rp.solidity, rp.eccentricity,
                      4 * np.pi * rp.area / per ** 2, rp.intensity_max, rp.intensity_mean,
                      rp.area * (maj / mnr)])
        masks.append(lbl == rp.label)
    return (np.array(feats, np.float64).reshape(-1, len(DESCR)), masks)


def build_blobs(han, gun, verbose=True):
    """All blobs at all thresholds, tagged with image index, threshold and gun-overlap label."""
    rows, img_idx, th_idx, is_gun = [], [], [], []
    gun_dil = [binary_dilation(g, disk(2)) if g.any() else g for g in gun]  # PSF slop ~2 cm
    t0 = time.time()
    for i in range(len(han)):
        th = tophat(han[i])
        for t, thr in enumerate(THRESHES):
            F, M = blob_table(han[i], th, thr)
            for j in range(len(F)):
                rows.append(F[j]); img_idx.append(i); th_idx.append(t)
                # a blob is a GUN blob if it covers >=30% of the (dilated) true footprint
                ov = (M[j] & gun_dil[i]).sum() / max(gun[i].sum(), 1) if gun[i].any() else 0.0
                is_gun.append(ov >= 0.30)
        if verbose and (i + 1) % 100 == 0:
            print(f"    blobs {i+1}/{len(han)} [{time.time()-t0:.0f}s]", flush=True)
    return (np.array(rows), np.array(img_idx), np.array(th_idx), np.array(is_gun))


# --------------------------------------------------- per-image feature families ----
def body_center_x(img):
    col = (img > -50).sum(1)
    return float((np.arange(NX) * col).sum() / max(col.sum(), 1))


def feats_bright(img):
    b = img[img > -60]
    if not len(b):
        b = img.ravel()
    return [np.percentile(b, 50), np.percentile(b, 90), np.percentile(b, 99),
            np.percentile(b, 99.9), float((img > -20).sum()), float((img > -30).sum()),
            float((img > -40).sum())]


F_BRIGHT = ["p50", "p90", "p99", "p999", "area_-20", "area_-30", "area_-40"]


def feats_sym(img):
    """Mirror about the BODY midline, on axis 0 (= x). Flipping axis 1 would mirror height."""
    c = int(round(body_center_x(img)))
    sh = NX // 2 - c
    a = np.roll(img, sh, axis=0)
    r = a - np.flip(a, axis=0)               # axis 0 == x == left/right.  NOT fliplr.
    r = r[:, :]
    return [float(r.max()), float(np.percentile(r, 99.9)), float(np.percentile(r, 99)),
            float((r > 6).sum()), float((r > 12).sum()), float(np.abs(r).mean())]


F_SYM = ["sym_max", "sym_p999", "sym_p99", "sym_area6", "sym_area12", "sym_absmean"]


def feats_shape(F, T, t):
    """Aggregate the blob descriptors of one image at one threshold."""
    f = F[T == t]
    if not len(f):
        return [0.0] * 10
    big = f[f[:, 0] >= 15]                   # blobs at least ~1 resolution cell
    aspect_big = big[:, 1].max() if len(big) else 0.0
    return [len(f), f[:, 0].max(), np.percentile(f[:, 0], 90), f[:, 0].mean(),
            f[:, 1].max(), aspect_big, f[:, 2].min(), f[:, 3].min(),
            f[:, 8].max(), float((f[:, 0] > 15).sum())]


F_SHAPE1 = ["n_blob", "area_max", "area_p90", "area_mean", "aspect_max", "aspect_big",
            "extent_min", "solid_min", "elong_max", "n_big"]


def image_features(han, B):
    """B = (rows, img_idx, th_idx, is_gun). Returns X, names."""
    rows, ii, tt, _ = B
    n = len(han)
    Xb, Xs, Xh = [], [], []
    for i in range(n):
        img = han[i]
        Xb.append(feats_bright(img))
        Xs.append(feats_sym(img))
        m = ii == i
        F, T = rows[m], tt[m]
        Xh.append(sum((feats_shape(F, T, t) for t in range(len(THRESHES))), []))
    names = (F_BRIGHT + F_SYM +
             [f"{c}@{thr:g}" for thr in THRESHES for c in F_SHAPE1])
    return np.hstack([np.array(Xb), np.array(Xs), np.array(Xh)]), names


# ------------------------------------------------------------------ CV -------------
def cv_auc(X, y, groups, model="lr", n_splits=5, seed=0):
    gkf = GroupKFold(n_splits=n_splits)
    aucs, oof = [], np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        if model == "lr":
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced"))
        else:
            clf = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced",
                                                 random_state=seed)
        Xtr = np.nan_to_num(X[tr]); Xte = np.nan_to_num(X[te])
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        oof[te] = p
        if len(set(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs)), oof


def cv_blobclf(B, y, groups, n_splits=5):
    """Per-blob classifier trained IN-FOLD, image score = max blob P(gun). Plan §7 step 5 recipe."""
    rows, ii, tt, isg = B
    gkf = GroupKFold(n_splits=n_splits)
    aucs, oof = [], np.full(len(y), np.nan)
    for tr, te in gkf.split(np.zeros((len(y), 1)), y, groups):
        trs, tes = set(tr.tolist()), set(te.tolist())
        mtr = np.array([i in trs for i in ii])
        mte = np.array([i in tes for i in ii])
        if isg[mtr].sum() < 10:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(np.nan_to_num(rows[mtr]), isg[mtr])
        p = clf.predict_proba(np.nan_to_num(rows[mte]))[:, 1]
        sub = ii[mte]
        for i in te:
            q = p[sub == i]
            oof[i] = q.max() if len(q) else 0.0
        aucs.append(roc_auc_score(y[te], oof[te]))
    return float(np.mean(aucs)), float(np.std(aucs)), oof


# ---------------------------------------------------------------- report -----------
def main():
    han, gun, meta = load()
    y = meta["y"].astype(int)
    body = meta["body"]
    site, stance, vis = meta["site"], meta["stance"], meta["visibility"].astype(float)
    print(f"dataset3: {len(y)} scans, {y.sum()} gun / {(y==0).sum()} no-gun, "
          f"{len(set(body))} bodies, {len(set(site))-1} carry sites", flush=True)
    print(f"gun footprint px: median {np.median(meta['gun_px'][y==1]):.0f}  "
          f"(range {meta['gun_px'][y==1].min()}-{meta['gun_px'][y==1].max()})", flush=True)

    print("\n[blob extraction]", flush=True)
    bcache = os.path.join(HERE, "ds3_blobs.npz")
    if os.path.exists(bcache):
        z = np.load(bcache)
        B = (z["rows"], z["ii"], z["tt"], z["isg"])
        print("  (reused ds3_blobs.npz)", flush=True)
    else:
        B = build_blobs(han, gun)
        np.savez_compressed(bcache, rows=B[0], ii=B[1], tt=B[2], isg=B[3])
    rows, ii, tt, isg = B
    if isg.sum() < 20:
        print(f"  !! only {isg.sum()} gun-tagged blobs — the gun rarely forms a blob at all. "
              f"Oracle below is unreliable.", flush=True)
    print(f"  {len(rows):,} blobs total, {isg.sum():,} tagged GUN "
          f"({isg.sum()/max((y==1).sum(),1):.2f} per positive image)", flush=True)

    # ---------------- PART 1: oracle ----------------
    print("\n=== PART 1 — ORACLE: gun blob vs body glint, by shape (uses ground truth) ===")
    print(f"{'descriptor':<12} {'AUC':>6}  {'gun mean':>10} {'body mean':>10}")
    for j, nm in enumerate(DESCR):
        a = roc_auc_score(isg, rows[:, j])
        print(f"{nm:<12} {max(a,1-a):>6.3f}  {rows[isg, j].mean():>10.3g} "
              f"{rows[~isg, j].mean():>10.3g}")
    aucs_multi = []
    gkf = GroupKFold(n_splits=5)
    gb = body[ii]
    for tr, te in gkf.split(rows, isg, gb):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,
                                                                 class_weight="balanced"))
        clf.fit(np.nan_to_num(rows[tr]), isg[tr])
        aucs_multi.append(roc_auc_score(isg[te], clf.predict_proba(np.nan_to_num(rows[te]))[:, 1]))
    print(f"\n  ALL descriptors, body-grouped CV : AUC {np.mean(aucs_multi):.3f} "
          f"+/- {np.std(aucs_multi):.3f}   <- is the shape signal there at all?")

    # ---------------- PART 2: blind detection ----------------
    print("\n=== PART 2 — BLIND per-image detection (body-grouped 5-fold CV) ===")
    X, names = image_features(han, B)
    nb, ns = len(F_BRIGHT), len(F_SYM)
    fams = {
        "brightness (control)": slice(0, nb),
        "symmetry": slice(nb, nb + ns),
        "blob shape": slice(nb + ns, X.shape[1]),
        "ALL": slice(0, X.shape[1]),
    }
    print(f"{'feature family':<22} {'LR AUC':>14} {'HGB AUC':>14}")
    oofs = {}
    for fam, sl in fams.items():
        a1, s1, o1 = cv_auc(X[:, sl], y, body, "lr")
        a2, s2, o2 = cv_auc(X[:, sl], y, body, "hgb")
        oofs[fam] = o2 if a2 > a1 else o1
        print(f"{fam:<22} {a1:>7.3f}+/-{s1:.3f} {a2:>7.3f}+/-{s2:.3f}")
    ab, sb, ob = cv_blobclf(B, y, body)
    oofs["per-blob clf -> max"] = ob
    print(f"{'per-blob clf -> max':<22} {ab:>7.3f}+/-{sb:.3f} {'-':>14}")

    print("\n  [shuffle-label controls — must be ~0.5 or the result above is noise]")
    for tag, seed in (("shuffle A", 1), ("shuffle B", 2)):
        ysh = np.random.default_rng(seed).permutation(y)
        a1, s1, _ = cv_auc(X, ysh, body, "lr")
        a2, s2, _ = cv_auc(X, ysh, body, "hgb")
        print(f"  {tag}: LR {a1:.3f}+/-{s1:.3f}   HGB {a2:.3f}+/-{s2:.3f}")

    # ---------------- PART 3: stratification ----------------
    best = max(fams, key=lambda f: roc_auc_score(y, np.nan_to_num(oofs[f])))
    if roc_auc_score(y, np.nan_to_num(ob)) > roc_auc_score(y, np.nan_to_num(oofs[best])):
        best = "per-blob clf -> max"
    sc = np.nan_to_num(oofs[best])
    print(f"\n=== PART 3 — STRATIFY (best family: {best}, pooled OOF AUC "
          f"{roc_auc_score(y, sc):.3f}) ===")
    neg = sc[y == 0]
    print(f"  no-gun score: median {np.median(neg):.3f}  p90 {np.percentile(neg,90):.3f}")
    print(f"\n  {'site':<16} {'n':>4} {'stance':>7} {'vis':>6} {'AUC vs no-gun':>14}")
    for st in sorted(set(site) - {"nothreat"}):
        for stc in ("front", "back"):
            m = (site == st) & (stance == stc)
            if m.sum() < 5:
                continue
            yy = np.r_[np.ones(m.sum()), np.zeros((y == 0).sum())]
            ss = np.r_[sc[m], neg]
            print(f"  {st:<16} {m.sum():>4} {stc:>7} {np.nanmean(vis[m]):>6.2f} "
                  f"{roc_auc_score(yy, ss):>14.3f}")
    for lo, hi, lab in ((0.0, 0.05, "occluded  vis<0.05"), (0.05, 0.25, "grazing   0.05-0.25"),
                        (0.25, 1.01, "visible   vis>0.25")):
        m = (y == 1) & (vis >= lo) & (vis < hi)
        if m.sum() < 5:
            continue
        yy = np.r_[np.ones(m.sum()), np.zeros((y == 0).sum())]
        print(f"  {lab:<24} n={m.sum():>3}   AUC {roc_auc_score(yy, np.r_[sc[m], neg]):.3f}")

    np.savez(os.path.join(OUT, "ds3_detect_scores.npz"), score=sc, y=y, site=site,
             stance=stance, vis=vis, body=body, best=best, X=X, names=np.array(names))
    figure(han, gun, meta, oofs, y, sc, best)


def figure(han, gun, meta, oofs, y, sc, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 9))
    ex = [i for i in range(len(y)) if y[i] == 1 and meta["visibility"][i] > 0.4][:3]
    ex += [i for i in range(len(y)) if y[i] == 0][:1]
    for c, i in enumerate(ex):
        ax = fig.add_subplot(2, 4, c + 1)
        ax.imshow(han[i].T, origin="lower", cmap="inferno", vmin=-45, vmax=0, aspect="equal")
        if gun[i].any():
            gx, gz = np.nonzero(gun[i])
            ax.plot(gx, gz, ",", color="#39f")
            ax.contour(gun[i].T.astype(float), levels=[0.5], colors="#39f", linewidths=1.2)
        ax.set_title(f"{meta['site'][i]} / {meta['stance'][i]}\n"
                     f"vis {meta['visibility'][i]:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(2, 4, 5)
    for fam, o in oofs.items():
        o = np.nan_to_num(o)
        fpr, tpr, _ = roc_curve(y, o)
        ax.plot(fpr, tpr, lw=1.6, label=f"{fam} ({roc_auc_score(y,o):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Pfa"); ax.set_ylabel("Pd"); ax.legend(fontsize=7); ax.set_title("ROC (OOF)")
    ax = fig.add_subplot(2, 4, 6)
    ax.hist(sc[y == 0], bins=25, alpha=0.6, label="no gun", color="#4a4a8a", density=True)
    ax.hist(sc[y == 1], bins=25, alpha=0.6, label="gun", color="#d94a3d", density=True)
    ax.legend(fontsize=8); ax.set_title(f"score dist ({best})", fontsize=9)
    ax = fig.add_subplot(2, 4, 7)
    v = meta["visibility"].astype(float)
    ax.scatter(v[y == 1], sc[y == 1], s=6, c="#d94a3d")
    ax.axhline(np.median(sc[y == 0]), color="#4a4a8a", ls="--", label="no-gun median")
    ax.set_xlabel("visibility"); ax.set_ylabel("score"); ax.legend(fontsize=8)
    ax.set_title("score vs gun visibility", fontsize=9)
    ax = fig.add_subplot(2, 4, 8)
    cd = meta["contrast_db"].astype(float)
    ax.scatter(cd[y == 1], sc[y == 1], s=6, c="#d94a3d")
    ax.set_xlabel("han_contrast_db"); ax.set_ylabel("score")
    ax.set_title("score vs gun/clutter contrast", fontsize=9)
    fig.suptitle("dataset3 — sim-internal gun detection (950 scans, 50 bodies, 2 m, Taylor, "
                 "specular/no-roughness)", fontsize=13)
    fig.tight_layout()
    p = os.path.join(OUT, "ds3_detect.png")
    fig.savefig(p, dpi=115, bbox_inches="tight")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
