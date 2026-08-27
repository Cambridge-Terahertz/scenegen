#!/usr/bin/env python3
"""Does the image task depend on having a CLEAN scan of the same body?

Two identical CNNs, same split, same schedule, differing only in input:
  RAW  = the reconstruction a real system would actually have.
  DIFF = the same scan minus the clean scan of the SAME body and view -- an oracle background
         subtraction that CANNOT exist on real data, where the person is never imaged without
         the threat.
The gap between them is the size of the advantage the fixed pose confers. A large gap means the
dataset is teaching "spot the deviation from a memorised body template" rather than "recognise
the object", which is the failure mode that would not survive transfer.

Excluded for fairness: the no-object baseline scans (their DIFF is identically zero, a pure
give-away) and the zero-visibility scans (no object signal exists in either input).
Body-disjoint folds throughout, so no body appears in train and test.
"""
import numpy as np, os, time, json, sys
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(int(os.environ.get("NT", "48")))
P = os.path.expanduser("~/probe")
EP = int(os.environ.get("EPOCHS", "8"))
NF = int(os.environ.get("FOLDS", "3"))

d = np.load(os.path.join(P, "ds6_img.npz"), allow_pickle=True)
Xr, Xd = d["Xr"], d["Xd"]
y = d["y"].astype(np.float32); body = d["body"].astype(str); obj = d["obj_name"] if "obj_name" in d.files else d["obj"]
obj = obj.astype(str)

m5 = np.load(os.path.join(P, "ds6_meta.npz"), allow_pickle=True)
# align by filename so the visibility mask is applied to the right rows
nm = d["name"].astype(str)
nv_by = {}
import glob
DATA = "/mnt/shared_drive/Jonah/synthetic_data/dataset6_2026-08-25"
fs = np.array([os.path.basename(f) for f in sorted(glob.glob(os.path.join(DATA,"**","*.npz"), recursive=True))])
assert len(fs) == len(m5["n_visible"])
nvmap = dict(zip(fs, m5["n_visible"].astype(float)))
nv = np.array([nvmap[n] for n in nm])

keep = (obj != "none") & (nv > 0)
print(f"kept {keep.sum()} of {len(keep)}  (dropped {(obj=='none').sum()} baseline, "
      f"{((obj!='none')&(nv==0)).sum()} zero-visibility)", flush=True)
Xr, Xd, y, body = Xr[keep], Xd[keep], y[keep], body[keep]
print(f"threat {int(y.sum())} / {len(y)} ({y.mean()*100:.1f}%)   bodies {len(set(body))}", flush=True)

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o),
                                            nn.ReLU(inplace=True), nn.MaxPool2d(2))
        s.f = nn.Sequential(blk(1,16), blk(16,32), blk(32,64), blk(64,128),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        s.h = nn.Linear(128, 1)
    def forward(s, x): return s.h(s.f(x)).squeeze(1)

def run(X, tag):
    aucs = []
    for k,(tr,te) in enumerate(GroupKFold(NF).split(X, y, body)):
        Xt = torch.from_numpy(X[tr]).float().div_(255).unsqueeze(1)
        Xe = torch.from_numpy(X[te]).float().div_(255).unsqueeze(1)
        yt = torch.from_numpy(y[tr]); ye = y[te]
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=256, shuffle=True, num_workers=0)
        net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        pos = yt.mean().item()
        lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-pos)/max(pos,1e-6)))
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, epochs=EP, steps_per_epoch=len(dl))
        net.train(); t0=time.time()
        for ep in range(EP):
            for xb, yb in dl:
                opt.zero_grad(); loss = lf(net(xb), yb); loss.backward(); opt.step(); sch.step()
        net.eval(); ps = []
        with torch.no_grad():
            for i in range(0, len(Xe), 512): ps.append(torch.sigmoid(net(Xe[i:i+512])).numpy())
        a = roc_auc_score(ye, np.concatenate(ps)); aucs.append(a)
        print(f"   {tag} fold {k}: AUC {a:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return float(np.mean(aucs)), float(np.std(aucs))

res = {}
for X, tag in ((Xr, "RAW  (what a real system has)"), (Xd, "DIFF (oracle clean-body subtraction)")):
    m, s = run(X, tag.split()[0]); res[tag] = [m, s]
    print(f"== {tag}: AUC {m:.4f} +/- {s:.4f}\n", flush=True)

print("== summary")
for k,(m,s) in res.items(): print(f"   {k:<40} {m:.4f} +/- {s:.4f}")
gap = res["DIFF (oracle clean-body subtraction)"][0] - res["RAW  (what a real system has)"][0]
print(f"   gap (oracle advantage): {gap:+.4f}")
json.dump(res, open(os.path.join(P,"ds6_bg.json"),"w"), indent=1)
