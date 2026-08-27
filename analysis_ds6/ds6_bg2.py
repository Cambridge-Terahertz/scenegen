#!/usr/bin/env python3
"""Re-run the RAW vs oracle-clean-subtracted comparison, saving out-of-fold predictions so the
result can be drawn as ROC curves rather than quoted as two numbers.

Identical model, split and schedule to ds6_bg.py -- only the prediction dump is added.
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

d = np.load(os.path.join(P, "ds6_img.npz"), allow_pickle=True)
Xr, Xd = d["Xr"], d["Xd"]
y = d["y"].astype(np.float32); body = d["body"].astype(str); obj = d["obj"].astype(str)
nm = d["name"].astype(str)
m5 = np.load(os.path.join(P, "ds6_meta.npz"), allow_pickle=True)
DATA = "/mnt/shared_drive/Jonah/synthetic_data/dataset6_2026-08-25"
fs = np.array([os.path.basename(f) for f in sorted(glob.glob(os.path.join(DATA,"**","*.npz"), recursive=True))])
nvmap = dict(zip(fs, m5["n_visible"].astype(float)))
nv = np.array([nvmap[n] for n in nm])

keep = (obj != "none") & (nv > 0)
Xr, Xd, y, body, nmk = Xr[keep], Xd[keep], y[keep], body[keep], nm[keep]
print(f"n={len(y)}  threat={int(y.sum())} ({y.mean()*100:.1f}%)  bodies={len(set(body))}", flush=True)

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o),
                                           nn.ReLU(inplace=True), nn.MaxPool2d(2))
        s.f = nn.Sequential(blk(1,16),blk(16,32),blk(32,64),blk(64,128),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        s.h = nn.Linear(128,1)
    def forward(s,x): return s.h(s.f(x)).squeeze(1)

out = {}
for X, tag in ((Xr,"raw"), (Xd,"diff")):
    p = np.zeros(len(y)); aucs=[]
    for k,(tr,te) in enumerate(GroupKFold(NF).split(X, y, body)):
        Xt = torch.from_numpy(X[tr]).float().div_(255).unsqueeze(1)
        Xe = torch.from_numpy(X[te]).float().div_(255).unsqueeze(1)
        yt = torch.from_numpy(y[tr])
        dl = DataLoader(TensorDataset(Xt,yt), batch_size=256, shuffle=True)
        net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        pos = yt.mean().item()
        lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-pos)/max(pos,1e-6)))
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, epochs=EP, steps_per_epoch=len(dl))
        net.train(); t0=time.time()
        for _ in range(EP):
            for xb,yb in dl:
                opt.zero_grad(); lf(net(xb), yb).backward(); opt.step(); sch.step()
        net.eval(); ps=[]
        with torch.no_grad():
            for i in range(0,len(Xe),512): ps.append(torch.sigmoid(net(Xe[i:i+512])).numpy())
        p[te] = np.concatenate(ps); a = roc_auc_score(y[te], p[te]); aucs.append(a)
        print(f"   {tag} fold {k}: AUC {a:.4f} ({time.time()-t0:.0f}s)", flush=True)
    out[tag] = p
    print(f"== {tag}: pooled AUC {roc_auc_score(y,p):.4f}  mean-of-folds {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}\n", flush=True)

np.savez(os.path.join(P,"ds6_bg_preds.npz"), y=y, raw=out["raw"], diff=out["diff"],
         body=body, name=nmk)
print("wrote ds6_bg_preds.npz", flush=True)
