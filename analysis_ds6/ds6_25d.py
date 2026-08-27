#!/usr/bin/env python3
"""Does the range axis matter?  2D projection vs 2.5D range stack, and what each owes to the
clean-body reference.

`han_db` -- and the 0.614 measured from it -- is a max-projection over range: the depth axis is
discarded entirely. `field_gated` keeps 21 bins at 5 mm around the surface, where an object proud
of the skin has a signature the projection destroys.

Four arms, identical CNN / folds / schedule, differing only in input. The 1-channel control is a
max-projection OF THE SAME SHELL, so "does range help" is not confounded by preprocessing:

  shell-2D        max over the 21 bins, 1 ch      the control
  shell-2.5D      the full 21-bin stack           does range recover what projection lost?
  shell-2D DIFF   control, clean subtracted       template advantage without range
  shell-2.5D DIFF stack, clean subtracted         does range REDUCE the template dependence?

That last comparison is the interesting one: if 2.5D RAW closes on 2.5D DIFF, depth is supplying
what background subtraction was supplying, and the fixed-pose exposure shrinks.
"""
import numpy as np, os, time, glob
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(int(os.environ.get("NT", "48")))
P = os.path.expanduser("~/probe")
EP, NF = int(os.environ.get("EPOCHS", "8")), int(os.environ.get("FOLDS", "3"))

d = np.load(os.path.join(P, "ds6_shell.npz"), allow_pickle=True)
Z = d["Z"]; y = d["y"].astype(np.float32); body = d["body"].astype(str)
obj = d["obj"].astype(str); nm = d["name"].astype(str); cidx = d["clean_idx"]

m5 = np.load(os.path.join(P, "ds6_meta.npz"), allow_pickle=True)
DATA = "/mnt/shared_drive/Jonah/synthetic_data/dataset6_2026-08-25"
fsn = np.array([os.path.basename(f) for f in sorted(glob.glob(os.path.join(DATA,"**","*.npz"), recursive=True))])
nvmap = dict(zip(fsn, m5["n_visible"].astype(float)))
nv = np.array([nvmap[n] for n in nm])

keep = (obj != "none") & (nv > 0)
ki = np.where(keep)[0]
print(f"kept {len(ki)} of {len(keep)}  threat {int(y[ki].sum())} ({y[ki].mean()*100:.1f}%)  "
      f"bodies {len(set(body[ki]))}", flush=True)

class Net(nn.Module):
    def __init__(s, cin):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o),
                                           nn.ReLU(inplace=True), nn.MaxPool2d(2))
        s.f = nn.Sequential(blk(cin,16),blk(16,32),blk(32,64),blk(64,128),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        s.h = nn.Linear(128,1)
    def forward(s,x): return s.h(s.f(x)).squeeze(1)

def make(idx, mode):
    """Build the input tensor for a set of rows. uint8 dB -> float in a sane range."""
    a = Z[idx].astype(np.float32) / 255.0                      # (n,21,H,W)
    if mode.endswith("diff"):
        a = a - Z[cidx[idx]].astype(np.float32) / 255.0
    if mode.startswith("2d"):
        a = a.max(axis=1, keepdims=True)                        # project over range
    return torch.from_numpy(a)

MODES = [("2d_raw",  "shell-2D        (projection, 1 ch)"),
         ("25d_raw", "shell-2.5D      (21-bin stack)"),
         ("2d_diff", "shell-2D DIFF   (projection, clean subtracted)"),
         ("25d_diff","shell-2.5D DIFF (stack, clean subtracted)")]
res = {}
for mode, label in MODES:
    p = np.zeros(len(ki)); aucs = []
    for k,(tr,te) in enumerate(GroupKFold(NF).split(ki, y[ki], body[ki])):
        Xt, Xe = make(ki[tr], mode), make(ki[te], mode)
        yt = torch.from_numpy(y[ki][tr])
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=256, shuffle=True)
        net = Net(Xt.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        pos = yt.mean().item()
        lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-pos)/max(pos,1e-6)))
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, epochs=EP, steps_per_epoch=len(dl))
        net.train(); t0 = time.time()
        for _ in range(EP):
            for xb, yb in dl:
                opt.zero_grad(); lf(net(xb), yb).backward(); opt.step(); sch.step()
        net.eval(); ps = []
        with torch.no_grad():
            for i in range(0, len(Xe), 512): ps.append(torch.sigmoid(net(Xe[i:i+512])).numpy())
        p[te] = np.concatenate(ps); a = roc_auc_score(y[ki][te], p[te]); aucs.append(a)
        print(f"   {mode} fold {k}: AUC {a:.4f} ({time.time()-t0:.0f}s)", flush=True)
        del Xt, Xe
    res[mode] = p
    print(f"== {label}: pooled {roc_auc_score(y[ki],p):.4f}  "
          f"mean {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}\n", flush=True)

np.savez(os.path.join(P,"ds6_25d_preds.npz"), y=y[ki], body=body[ki], name=nm[ki],
         **{k:v for k,v in res.items()})
print("== summary")
for mode, label in MODES:
    print(f"   {label}  {roc_auc_score(y[ki], res[mode]):.4f}")
g2 = roc_auc_score(y[ki],res["2d_diff"]) - roc_auc_score(y[ki],res["2d_raw"])
g25 = roc_auc_score(y[ki],res["25d_diff"]) - roc_auc_score(y[ki],res["25d_raw"])
print(f"   clean-body advantage:  2D {g2:+.4f}   2.5D {g25:+.4f}")
