"""Summary figure: 2D max-projection vs 3D range gate on dataset3."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
import ds3_detect as D
from sklearn.metrics import roc_curve, roc_auc_score

han, gun, meta = D.load()
maps = np.load(os.path.join(HERE, "ds3_cache3d.npz"))["maps"].astype(np.float32)
z = np.load(os.path.join(HERE, "ds3_detect_scores.npz"), allow_pickle=True)
X2 = z["X"]; y = z["y"].astype(int); site = z["site"]; body = z["body"]
X3 = np.load(os.path.join(HERE, "ds3_X3.npy")); XC = np.hstack([X2, X3])

fig = plt.figure(figsize=(16, 9))
# --- top row: one torso-carry scan, 2D vs range-spread map
i = int(np.where((site == "belly") & (y == 1))[0][0])
valid = maps[i, 0] > -900
panels = [(han[i], "2D  han_db  (max-projected)", -45, 0, "inferno"),
          (np.where(valid, maps[i, 1], np.nan), "3D  range SPREAD (mm)", 0, 40, "viridis"),
          (np.where(valid, maps[i, 2], np.nan), "3D  # range peaks", 0, 6, "viridis")]
for c, (M, t, lo, hi, cm) in enumerate(panels):
    ax = fig.add_subplot(2, 3, c + 1)
    im = ax.imshow(M.T, origin="lower", cmap=cm, vmin=lo, vmax=hi, aspect="equal")
    ax.contour(gun[i].T.astype(float), levels=[0.5], colors="#39f", linewidths=1.4)
    ax.set_title(t + "\nblue = true gun footprint", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, fraction=0.046)

# --- ROC by feature set
ax = fig.add_subplot(2, 3, 4)
for nm, XX in (("2D only", X2), ("3D range only", X3), ("2D + 3D", XC)):
    _, _, o = D.cv_auc(XX, y, body, "lr")
    fpr, tpr, _ = roc_curve(y, np.nan_to_num(o))
    ax.plot(fpr, tpr, lw=1.8, label=f"{nm} ({roc_auc_score(y, np.nan_to_num(o)):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.legend(fontsize=9)
ax.set_xlabel("Pfa"); ax.set_ylabel("Pd"); ax.set_title("ROC, all 950 scans (OOF)", fontsize=11)

# --- AUC by subset
ax = fig.add_subplot(2, 3, 5)
subs = ["all\nsites", "ankle\nonly", "torso\nonly", "midline\nonly"]
v2 = [0.654, 0.873, 0.575, 0.588]; v3 = [0.749, 0.566, 0.924, 0.876]
xx = np.arange(4); w = 0.38
ax.bar(xx - w/2, v2, w, label="2D max-projection", color="#d94a3d")
ax.bar(xx + w/2, v3, w, label="3D range gate", color="#3a7abd")
ax.axhline(0.55, color="k", ls="--", lw=.9, label="shuffle null p95")
ax.set_xticks(xx); ax.set_xticklabels(subs, fontsize=9); ax.set_ylim(0.4, 1.0)
ax.set_ylabel("AUC"); ax.legend(fontsize=8)
ax.set_title("2D and 3D are COMPLEMENTARY by carry site", fontsize=11)

# --- gun-blob rank
ax = fig.add_subplot(2, 3, 6)
lbl = ["2D only", "3D only", "2D+3D"]
t1 = [22.7, 21.0, 42.5]; t5 = [42.3, 61.2, 80.3]; t20 = [75.6, 86.3, 96.9]
xx = np.arange(3)
ax.bar(xx - .25, t1, .25, label="top-1", color="#2c4a7c")
ax.bar(xx, t5, .25, label="top-5", color="#5b8cc4")
ax.bar(xx + .25, t20, .25, label="top-20", color="#a9c8e8")
ax.set_xticks(xx); ax.set_xticklabels(lbl); ax.set_ylabel("% of images")
ax.legend(fontsize=8)
ax.set_title("Is the true gun blob ranked highly?\n(out of ~346 blobs/image)", fontsize=11)
fig.suptitle("dataset3 — the stored range gate, not the 2D max-projection, is what detects a "
             "torso-carried gun  (950 scans, 50 bodies, 2 m, specular/no-roughness)", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "ds3_2d_vs_3d.png"), dpi=115, bbox_inches="tight")
print("saved -> ds3_2d_vs_3d.png")
