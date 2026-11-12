#!/usr/bin/env python3
"""Benchmark forward-model variants on the captured scene, fp32:
  baseline  = raforward.py (untouched)
  v1        = ph/TWO_PI -> ph*INV_TWO_PI   (Fable idea #1, register-neutral)
  v3        = v1 + --use_fast_math          (Fable idea #3)
Confirms accuracy vs Scott's release (A_sig) is preserved. Run in `ra` env,
cwd = ~/Documents/code/model_gen/python.
"""
import time, json
import numpy as np, cupy as cp
import raforward as R0
import raforward_v1 as R1
import raforward_v3 as R3

A = np.load("/home/jonah/scott_release/compare/A_capture.npz")
targ = A["targ"].astype(np.float64); s = A["s"].astype(np.float64).ravel()
txs = A["txs"].astype(np.float64); rx0 = A["rx0"].astype(np.float64)
f = np.imag(A["k"].ravel()) * 299792458.0 / (2*np.pi)
kB = 2*np.pi*f/299792458.0
A_sig = A["A_sig"]
n_eval = targ.shape[0]*rx0.shape[0]*f.size
print(f"scene: ntarg={targ.shape[0]} nrx={rx0.shape[0]} nf={f.size}  n_eval={n_eval/1e9:.1f}e9")

def _sync(): cp.cuda.Stream.null.synchronize()
def relerr(a,b): return float(np.max(np.abs(a-b))/np.max(np.abs(b)))

def bench(mod, reps=7):
    scene = mod.Scene(q=targ, s=s, tx=txs, rx=rx0, k=kB)
    out = mod.forward(scene, backend="gpu", precision="fp32"); _sync()  # warmup+compile
    ts=[]
    for _ in range(reps):
        _sync(); t0=time.perf_counter()
        out = mod.forward(scene, backend="gpu", precision="fp32"); _sync()
        ts.append(time.perf_counter()-t0)
    return np.asarray(out), min(ts)

res={}
for name, mod in [("baseline",R0),("v1_div2mul",R1),("v3_v1+fastmath",R3)]:
    sig, t = bench(mod)
    res[name] = {"min_s": t, "Geval_s": n_eval/t/1e9,
                 "relerr_vs_release": relerr(sig, A_sig)}
    print(f"{name:16s} min={t:.4f}s  {n_eval/t/1e9:6.1f} Geval/s  relerr_vs_release={relerr(sig,A_sig):.2e}")

base=res["baseline"]["min_s"]
for name in ("v1_div2mul","v3_v1+fastmath"):
    res[name]["speedup_vs_baseline"] = base/res[name]["min_s"]
    print(f"  {name}: {base/res[name]['min_s']:.3f}x vs baseline")

with open("/home/jonah/scott_release/compare/fm_opt_results.json","w") as fh:
    json.dump(res, fh, indent=2)
print("wrote fm_opt_results.json")
