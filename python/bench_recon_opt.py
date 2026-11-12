#!/usr/bin/env python3
"""Benchmark baseline recon.py vs optimized recon_opt.py (fast-math + hoisted
reciprocals) on the identical captured scene, and confirm the numeric match to
Scott's release survives. Run in `ra` env, cwd = ~/Documents/code/model_gen/python.
"""
import time, json
import numpy as np, cupy as cp
import recon as R          # baseline (untouched)
import recon_opt as Ro     # optimized copy

C = "/home/jonah/scott_release/compare"
A = np.load(f"{C}/A_capture.npz")
B = np.load(f"{C}/B_outputs.npz")
xs, ys, zs = A["xs"].ravel(), A["ys"].ravel(), A["zs"].ravel()
A_recon = A["A_recon"]                 # release recon (reference)
sig = B["B_sig32"]                     # port FM fp32 output (same input to both recons)
txs = A["txs"].astype(np.float64); rx0 = A["rx0"].astype(np.float64)
f = np.imag(A["k"].ravel()) * 299792458.0 / (2*np.pi)
print(f"scene: {xs.size*ys.size*zs.size} voxels, nrx={rx0.shape[0]}, nf={f.size}")

def _sync(): cp.cuda.Stream.null.synchronize()

def timed(mod, grid, reps=3):
    kw = dict(tx=txs, rx=rx0, f=f, grid=grid, freq_window="none",
              backend="gpu", precision="fp32")
    img = mod.backproject(sig, **kw); _sync()      # warmup + compile
    ts = []
    for _ in range(reps):
        _sync(); t0 = time.perf_counter()
        img = mod.backproject(sig, **kw); _sync()
        ts.append(time.perf_counter() - t0)
    return np.asarray(img), min(ts)

def cplx_corr(a, b):
    a = a.ravel().astype(np.complex128); b = b.ravel().astype(np.complex128)
    return float(np.abs(np.vdot(a, b)) / (np.linalg.norm(a)*np.linalg.norm(b)))

img_base, t_base = timed(R,  R.ReconGrid(xs=xs, ys=ys, zs=zs))
img_opt,  t_opt  = timed(Ro, Ro.ReconGrid(xs=xs, ys=ys, zs=zs))

res = {
    "t_baseline_s": t_base,
    "t_optimized_s": t_opt,
    "speedup_opt_over_baseline": t_base / t_opt,
    "corr_opt_vs_baseline": cplx_corr(img_opt, img_base),   # fast-math drift
    "corr_baseline_vs_release": cplx_corr(img_base, A_recon),
    "corr_opt_vs_release": cplx_corr(img_opt, A_recon),      # must stay ~0.99999
    "peak_idx_opt": [int(i) for i in np.unravel_index(np.argmax(np.abs(img_opt)), img_opt.shape)],
    "peak_idx_release": [int(i) for i in np.unravel_index(np.argmax(np.abs(A_recon)), A_recon.shape)],
}
print(json.dumps(res, indent=2))
with open(f"{C}/recon_opt_results.json", "w") as fh:
    json.dump(res, fh, indent=2)
print("wrote", f"{C}/recon_opt_results.json")
