#!/usr/bin/env python3
"""Forward-model variants at two scales, fp32:
  baseline = raforward.py
  v3       = div->mul + fast-math  (confirmed 1.44x at small scale)
  vrec     = phase-recursion (2 sincos anchors + rotations) + v3   [PROTOTYPE]
Small scale = captured release scene (parity vs release). Large = 3M synthetic
(exceeds L2 -> memory regime; parity vrec vs baseline). Run in `ra`, cwd=python.
"""
import time, json
import numpy as np, cupy as cp
import raforward as R0, raforward_v3 as R3, raforward_vrec as RR
C0 = 299792458.0

A = np.load("/home/jonah/scott_release/compare/A_capture.npz")
targ = A["targ"].astype(np.float64); s = A["s"].astype(np.float64).ravel()
txs = A["txs"].astype(np.float64); rx0 = A["rx0"].astype(np.float64)
f = np.imag(A["k"].ravel()) * C0 / (2*np.pi); kB = 2*np.pi*f/C0
A_sig = A["A_sig"]

def sync(): cp.cuda.Stream.null.synchronize()
def relerr(a,b): return float(np.max(np.abs(a-b))/np.max(np.abs(b)))

def bench(mod, scene, reps):
    out = mod.forward(scene, backend="gpu", precision="fp32"); sync()   # warmup+compile
    ts=[]
    for _ in range(reps):
        sync(); t0=time.perf_counter(); out=mod.forward(scene, backend="gpu", precision="fp32"); sync()
        ts.append(time.perf_counter()-t0)
    return np.asarray(out), min(ts)

variants = [("baseline",R0),("v3",R3),("vrec",RR)]
out = {"small": {}, "large": {}}

# ---- SMALL: captured scene (41762 pts), parity vs release ----
ne_s = targ.shape[0]*rx0.shape[0]*f.size
print(f"== SMALL: ntarg={targ.shape[0]} (L2-resident) ==")
sigs={}
for name,mod in variants:
    sc = mod.Scene(q=targ, s=s, tx=txs, rx=rx0, k=kB)
    sig,t = bench(mod, sc, reps=5); sigs[name]=sig
    out["small"][name] = {"t":t, "Geval_s":ne_s/t/1e9, "relerr_vs_release":relerr(sig,A_sig)}
    print(f"  {name:9s} {t:.4f}s  {ne_s/t/1e9:6.1f} Geval/s  relerr_vs_release={relerr(sig,A_sig):.2e}")
print(f"  vrec vs baseline relerr = {relerr(sigs['vrec'], sigs['baseline']):.2e}  (parity gate)")
out["small"]["vrec_vs_baseline_relerr"] = relerr(sigs['vrec'], sigs['baseline'])
b=out["small"]["baseline"]["t"]
for n in ("v3","vrec"): print(f"    {n}: {b/out['small'][n]['t']:.3f}x vs baseline")

# ---- LARGE: 3M synthetic (exceeds L2), parity vrec vs baseline ----
NL=3_000_000; ne_l=NL*rx0.shape[0]*f.size
print(f"\n== LARGE: ntarg={NL} (exceeds 4MB L2 -> memory regime) ==")
sigsL={}
for name,mod in variants:
    sc = mod.Scene.demo(ntarg=NL, nrx=16900, nf=128, seed=0)   # deterministic, identical across modules
    sig,t = bench(mod, sc, reps=2); sigsL[name]=sig
    out["large"][name] = {"t":t, "Geval_s":ne_l/t/1e9}
    print(f"  {name:9s} {t:7.2f}s  {ne_l/t/1e9:6.1f} Geval/s")
out["large"]["vrec_vs_baseline_relerr"] = relerr(sigsL['vrec'], sigsL['baseline'])
print(f"  vrec vs baseline relerr = {out['large']['vrec_vs_baseline_relerr']:.2e}  (parity gate)")
bL=out["large"]["baseline"]["t"]
for n in ("v3","vrec"): print(f"    {n}: {bL/out['large'][n]['t']:.3f}x vs baseline")

with open("/home/jonah/scott_release/compare/fm_scale_results.json","w") as fh:
    json.dump(out, fh, indent=2)
print("\nwrote fm_scale_results.json")
