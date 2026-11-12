#!/usr/bin/env python3
"""Drive the CuPy port on the EXACT arrays captured from Scott's release,
then compare forward-model + recon numerically and time both on the A4000.

Run in the `ra` conda env, cwd = ~/Documents/code/model_gen/python.
"""
import time, json
import numpy as np
import cupy as cp
import raforward as rf
import recon as R

C0 = 299792458.0
CAP = "/home/jonah/scott_release/compare/A_capture.npz"
d = np.load(CAP)
targ = d["targ"].astype(np.float64)
txs  = d["txs"].astype(np.float64)          # (ntx,3)
rx0  = d["rx0"].astype(np.float64)          # (nrx,3)
s    = d["s"].astype(np.float64).ravel()
k_A  = d["k"].ravel()                        # complex 1j*2pi f/C
A_sig = d["A_sig"]                           # (nf, nrx) complex
xs, ys, zs = d["xs"].ravel(), d["ys"].ravel(), d["zs"].ravel()
A_recon = d["A_recon"]                        # (nx,ny,nz) complex

# recover real frequencies from A's k = 1j*2*pi*f/C  ->  f = Im(k)*C/2pi
f = np.imag(k_A) * C0 / (2 * np.pi)
kB = 2 * np.pi * f / C0                        # port wants real k = 2pi f/c
nf, nrx = A_sig.shape
print(f"[B] scene: ntarg={targ.shape[0]} nrx={nrx} nf={nf} "
      f"grid={(xs.size,ys.size,zs.size)} = {xs.size*ys.size*zs.size} voxels")

scene = rf.Scene(q=targ, s=s, tx=txs, rx=rx0, k=kB)

def _sync():
    cp.cuda.Stream.null.synchronize()

def timed_forward(prec, reps=5):
    out = rf.forward(scene, backend="gpu", precision=prec)  # warmup + kernel compile
    _sync()
    ts = []
    for _ in range(reps):
        _sync(); t0 = time.perf_counter()
        out = rf.forward(scene, backend="gpu", precision=prec)
        _sync(); ts.append(time.perf_counter() - t0)
    return np.asarray(out), min(ts)

def relerr(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))

def cplx_corr(a, b):
    a = a.ravel().astype(np.complex128); b = b.ravel().astype(np.complex128)
    return float(np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ---------- Forward-model comparison ----------
B_sig32, t_fm32 = timed_forward("fp32")
B_sig64, t_fm64 = timed_forward("fp64")
fm_res = {
    "A_fm_wall_s_incl_io": float(d["fm_wall_s"]) if "fm_wall_s" in d else None,
    "B_fm_fp32_kernel_s": t_fm32,
    "B_fm_fp64_kernel_s": t_fm64,
    "relerr_Bfp32_vs_A": relerr(B_sig32, A_sig),
    "relerr_Bfp64_vs_A": relerr(B_sig64, A_sig),
    "corr_Bfp32_vs_A": cplx_corr(B_sig32, A_sig),
    "corr_Bfp64_vs_A": cplx_corr(B_sig64, A_sig),
    "A_sig_absmax": float(np.max(np.abs(A_sig))),
    "B_sig32_absmax": float(np.max(np.abs(B_sig32))),
}
print("[B] forward-model:", json.dumps(fm_res, indent=2))

# ---------- Recon comparison (same grid, freq_window='none' to match A's DFT) ----------
grid = R.ReconGrid(xs=xs, ys=ys, zs=zs)

def timed_recon(sig, tag):
    _ = R.backproject(sig, tx=txs, rx=rx0, f=f, grid=grid,
                      freq_window="none", backend="gpu", precision="fp32")  # warmup
    _sync(); t0 = time.perf_counter()
    img = R.backproject(sig, tx=txs, rx=rx0, f=f, grid=grid,
                        freq_window="none", backend="gpu", precision="fp32")
    _sync(); dt = time.perf_counter() - t0
    print(f"[B] recon ({tag}) wall={dt:.2f}s shape={img.shape}")
    return np.asarray(img), dt

# end-to-end: port's own FM (fp32, production) -> port recon
B_img_e2e, t_rec_e2e = timed_recon(B_sig32, "port FM fp32 -> port recon")
# isolate the recon algorithm: feed A's OWN sig into the port recon
B_img_fromA, t_rec_fromA = timed_recon(A_sig, "A FM -> port recon")

def img_metrics(A, B):
    Aa = np.abs(A.ravel().astype(np.float64)); Bb = np.abs(B.ravel().astype(np.float64))
    corr_mag = float(np.corrcoef(Aa, Bb)[0, 1])
    # peak location agreement
    pa = np.unravel_index(np.argmax(np.abs(A)), A.shape)
    pb = np.unravel_index(np.argmax(np.abs(B)), B.shape)
    dx = (xs[pa[0]] - xs[pb[0]]); dy = (ys[pa[1]] - ys[pb[1]]); dz = (zs[pa[2]] - zs[pb[2]])
    peak_dist_m = float(np.sqrt(dx*dx + dy*dy + dz*dz))
    return {"corr_complex": cplx_corr(A, B), "corr_magnitude": corr_mag,
            "A_peak_idx": list(map(int, pa)), "B_peak_idx": list(map(int, pb)),
            "peak_dist_m": peak_dist_m}

rec_res = {
    "A_recon_shape": list(A_recon.shape),
    "B_recon_shape": list(B_img_e2e.shape),
    "B_recon_wall_s_e2e": t_rec_e2e,
    "B_recon_wall_s_fromAsig": t_rec_fromA,
    "e2e_vs_A(port FM+recon vs release FM+recon)": img_metrics(A_recon, B_img_e2e),
    "recon_algo_only(A_sig->port recon vs A_sig->release recon)": img_metrics(A_recon, B_img_fromA),
}
print("[B] recon:", json.dumps(rec_res, indent=2))

# A timing from timing.mat (seconds), if present
A_timing = {k[len("timing_"):]: d[k].tolist() for k in d.files if k.startswith("timing_")}

summary = {"forward_model": fm_res, "recon": rec_res, "A_timing_mat": A_timing,
           "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}
with open("/home/jonah/scott_release/compare/comparison_results.json", "w") as fh:
    json.dump(summary, fh, indent=2)
np.savez("/home/jonah/scott_release/compare/B_outputs.npz",
         B_sig32=B_sig32, B_sig64=B_sig64, B_img_e2e=B_img_e2e, B_img_fromA=B_img_fromA)
print("\n[B] wrote comparison_results.json + B_outputs.npz")
print("[B] DONE")
