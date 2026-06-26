"""
CPU reference for the reflectarray frequency-domain forward model.

Mirrors the live CUDA kernel
    cpp/ForwardModelCudaProject/forward_model_mex.cu
(invoked at simulator_mixamo_flow.m:528 as ForwardModelCudaProject(...))

Physics (bistatic, scalar, first-order Born, point scatterers):

    sig[f, i] = sum_tx sum_t  s_t / (r1 * r2) * exp(-j k_f (r1 + r2))
    r1 = |TX_tx - q_t|   (TX leg; independent of rx and freq)
    r2 = |RX_i  - q_t|   (RX leg; independent of freq)
    k_f = 2*pi*f/c

Implementations:
  - forward_brute : literal transcription of the kernel = the SPEC.
                    phase32=True models the GPU's genuine single precision.
  - forward_fast  : the restructured math for the new kernel
                    (Step 2: precompute r1 once, reuse r2 across freq;
                     Step 3: phase recursion over frequency).
                    Optional mitigations: r_ref (range reference) and reanchor.

Findings this reference established on a CPU (no GPU needed):

  1. The restructure is EXACT in float64 (parity ~1e-12). The new kernel's math
     is correct.

  2. float32 has TWO distinct error sources, each ~1e-3, each with a fix:
       (A) LARGE ABSOLUTE PHASE. k*(r1+r2) ~ 25000 rad at THz/multi-metre range;
           float32 resolves that to only ~3e-3 rad. This hits the CURRENT kernel
           (it computes phase=-kf*(r1+r2) in float32). Fix: range-reference the
           phase (subtract a constant r_ref, re-apply exp(-j k r_ref) exactly).
       (B) PHASE-RECURSION DRIFT. The Step-3 recursion w*=rot compounds float32
           rounding over nf steps. Fix: re-anchor every K steps (recompute exp),
           or skip the recursion and just call sincos per freq (the GPU's __sincosf
           is cheap, so Step 3 may not be worth its accuracy cost).

  This does NOT validate against an actual MATLAB run (needs MATLAB + GPU).
  forward_brute IS the spec to diff that run against.
"""

from __future__ import annotations
import numpy as np

C0 = 299_792_458.0


def _ref_phase(sig, k, r_ref, dtype):
    """Re-apply the exact reference phase exp(-j k r_ref) per frequency."""
    if r_ref:
        sig *= np.exp(-1j * np.asarray(k, np.float64) * r_ref).astype(dtype)[:, None]
    return sig


def forward_brute(TARG, TXS, RX0, s, k, dtype=np.complex128, phase32=False, r_ref=0.0):
    """Brute-force forward model = the spec. Mirrors forward_model_mex.cu.

    phase32=True : compute distances/phase/trig/accumulation in single precision
                   to faithfully model the GPU kernel's float32 behaviour.
    r_ref        : constant reference range subtracted before forming the phase
                   (re-applied exactly afterward). Keeps the float32 phase small.
    """
    rdt = np.float32 if phase32 else np.float64
    cdt = np.complex64 if phase32 else np.complex128
    TARG = np.asarray(TARG, np.float64)
    TXS = np.atleast_2d(np.asarray(TXS, np.float64))
    RX0 = np.asarray(RX0, np.float64)
    s = np.asarray(s, np.float64)
    k = np.asarray(k, np.float64)
    nf, nrx = k.shape[0], RX0.shape[0]
    krt = k.astype(rdt)

    sig = np.zeros((nf, nrx), dtype=cdt)
    for tx in TXS:
        r1 = np.linalg.norm(TARG - tx, axis=1).astype(rdt)
        for i in range(nrx):
            r2 = np.linalg.norm(TARG - RX0[i], axis=1).astype(rdt)
            amp = (s.astype(rdt) / (r1 * r2)).astype(cdt)
            rs = (r1 + r2 - rdt(r_ref))
            ang = np.outer(krt, rs).astype(rdt)         # (nf,ntarg) single-precision
            ph = np.empty(ang.shape, cdt)
            ph.real = np.cos(ang)                        # genuine rdt trig
            ph.imag = -np.sin(ang)
            sig[:, i] += ph @ amp                       # rdt-precision reduction
    sig = _ref_phase(sig, k, r_ref, cdt)
    return sig.astype(dtype)


def forward_fast(TARG, TXS, RX0, s, k, dtype=np.complex128, r_ref=0.0, reanchor=0):
    """Restructured forward model: precompute r1, reuse r2, phase recursion.

    r_ref    : range reference (fixes float32 error source A; no-op in float64).
    reanchor : if >0, recompute the phasor exactly every `reanchor` frequency
               steps to bound recursion drift (fixes float32 error source B).
               reanchor=1 is equivalent to no recursion (exp per freq).
    """
    # KEY: distances and the phase argument are always computed in float64
    # (THz * multi-metre range => float32 distances lose ~1e-3 rad of phase).
    # `dtype` controls only the phasor STORAGE and the recursion accumulation.
    TARG = np.asarray(TARG, np.float64)
    TXS = np.atleast_2d(np.asarray(TXS, np.float64))
    RX0 = np.asarray(RX0, np.float64)
    s = np.asarray(s, np.float64)
    k = np.asarray(k, np.float64)
    nf, nrx = k.shape[0], RX0.shape[0]

    if nf > 1:
        dk = (k[-1] - k[0]) / (nf - 1)
        if not np.allclose(np.diff(k), dk, rtol=1e-6, atol=0):
            raise ValueError("forward_fast requires linearly spaced k")
    else:
        dk = 0.0
    k0 = k[0]

    sig = np.zeros((nf, nrx), dtype=dtype)
    for tx in TXS:
        r1 = np.linalg.norm(TARG - tx, axis=1)                  # Step 2: once per target (f64)
        for i in range(nrx):
            r2 = np.linalg.norm(TARG - RX0[i], axis=1)          # once per (target,rx) (f64)
            amp = s / (r1 * r2)
            rs = r1 + r2 - r_ref                                # f64; r_ref optional
            w = (amp * np.exp(-1j * k0 * rs)).astype(dtype)     # base phasor (f64 -> store)
            rot = np.exp(-1j * dk * rs).astype(dtype)           # increment
            for m in range(nf):                                 # Step 3: recursion (in dtype)
                if reanchor and m > 0 and m % reanchor == 0:    # re-anchor to bound f32 drift
                    w = (amp * np.exp(-1j * (k0 + m * dk) * rs)).astype(dtype)
                sig[m, i] += w.sum()
                w = w * rot
    sig = _ref_phase(sig, k, r_ref, dtype)
    return sig


# ---------------------------------------------------------------------------
def make_scene(ntarg=200, nrx=64, nf=128, ntx=1, seed=0,
               f1=275e9, f2=305e9, array_size=0.067, standoff=2.0):
    rng = np.random.default_rng(seed)
    q = np.column_stack([
        rng.uniform(-0.25, 0.25, ntarg),
        standoff + rng.uniform(-0.35, 0.35, ntarg),
        rng.uniform(-0.5, 0.5, ntarg),
    ])
    s = rng.uniform(0.5, 1.0, ntarg)
    side = int(round(np.sqrt(nrx)))
    g = np.linspace(-array_size / 2, array_size / 2, side)
    xx, zz = np.meshgrid(g, g, indexing="ij")
    RX0 = np.column_stack([xx.ravel(), np.zeros(side * side), zz.ravel()])
    TXS = np.tile([0.15, 0.0, 0.0], (ntx, 1))
    k = 2 * np.pi * np.linspace(f1, f2, nf) / C0
    return q, TXS, RX0, s, k


def scene_r_ref(q, TXS, RX0):
    """A good constant reference range: mean round-trip to scene/array centres."""
    return float(np.linalg.norm(q - TXS[0], axis=1).mean()
                 + np.linalg.norm(q - RX0.mean(0), axis=1).mean())


def relerr(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


# ---------------------------------------------------------------------------
# Tests (pytest-discoverable)
# ---------------------------------------------------------------------------
def test_parity_float64():
    q, TXS, RX0, s, k = make_scene(ntarg=150, nrx=49, nf=64)
    a = forward_brute(q, TXS, RX0, s, k)
    b = forward_fast(q, TXS, RX0, s, k)
    assert relerr(b, a) < 1e-9, relerr(b, a)


def test_parity_multitx():
    q, TXS, RX0, s, k = make_scene(ntarg=120, nrx=36, nf=32, ntx=3, seed=3)
    a = forward_brute(q, TXS, RX0, s, k)
    b = forward_fast(q, TXS, RX0, s, k)
    assert relerr(b, a) < 1e-9, relerr(b, a)


def test_single_frequency():
    q, TXS, RX0, s, k = make_scene(ntarg=80, nrx=25, nf=1)
    a = forward_brute(q, TXS, RX0, s, k)
    b = forward_fast(q, TXS, RX0, s, k)
    assert relerr(b, a) < 1e-9, relerr(b, a)


def test_current_kernel_float32_is_inaccurate():
    # pure single precision (current kernel model) has ~1e-3 phase error from
    # representing multi-metre distances in float32
    q, TXS, RX0, s, k = make_scene(ntarg=150, nrx=49, nf=64)
    ref = forward_brute(q, TXS, RX0, s, k)                       # float64 truth
    cur = forward_brute(q, TXS, RX0, s, k, phase32=True)         # all-float32
    assert relerr(cur, ref) > 1e-4                               # demonstrably worse


def test_restructured_float32_is_accurate():
    # forward_fast keeps distances+phase in float64; with reanchor the float32
    # accumulation is accurate to ~1e-6 -- far better than the current kernel
    q, TXS, RX0, s, k = make_scene(ntarg=150, nrx=49, nf=64)
    ref = forward_brute(q, TXS, RX0, s, k)
    cur = forward_brute(q, TXS, RX0, s, k, phase32=True)
    good = forward_fast(q, TXS, RX0, s, k, dtype=np.complex64, reanchor=8)
    assert relerr(good, ref) < relerr(cur, ref) / 20


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    q, TXS, RX0, s, k = make_scene(ntarg=400, nrx=121, nf=128)
    ntarg, nrx, nf = q.shape[0], RX0.shape[0], k.shape[0]
    r_ref = scene_r_ref(q, TXS, RX0)
    print(f"scene: ntarg={ntarg}  nrx={nrx}  nf={nf}  ntx={TXS.shape[0]}  r_ref={r_ref:.3f} m")

    ref = forward_brute(q, TXS, RX0, s, k)                                   # float64 truth
    fast = forward_fast(q, TXS, RX0, s, k)

    print("\n1) restructure is EXACT in float64:")
    print(f"     forward_fast vs brute : relerr = {relerr(fast, ref):.2e}")

    pos_err = (np.linalg.norm(q-TXS[0], axis=1).mean()) * 2**-23
    print(f"\n2) float32 accuracy (root cause = multi-metre distances; ~{pos_err:.0e} m error -> "
          f"~{2*np.pi*290e9/C0*pos_err:.0e} rad):")
    cur          = forward_brute(q, TXS, RX0, s, k, phase32=True)             # current kernel: all f32
    cur_pos32    = forward_brute(q, TXS, RX0, s, k)                           # (truth; for contrast only)
    fast32       = forward_fast(q, TXS, RX0, s, k, dtype=np.complex64)        # f64 dist, f32 accum, recursion
    fast32_anc   = forward_fast(q, TXS, RX0, s, k, dtype=np.complex64, reanchor=8)
    print(f"   current kernel model   (all float32)           : {relerr(cur, ref):.2e}   <-- accuracy floor today")
    print(f"   restructured (f64 dist, f32 accum, recursion)  : {relerr(fast32, ref):.2e}   (residual = recursion drift)")
    print(f"   restructured + reanchor(8)                     : {relerr(fast32_anc, ref):.2e}   <-- ~20x better than current")

    print("\n3) transcendental count (GPU-independent part of the win):")
    print(f"     brute ~ nf*ntarg = {nf*ntarg:,} exp/rx   vs   fast ~ 2*ntarg = {2*ntarg:,} exp/rx   (~{nf//2}x fewer)")

    print("\nOK" if relerr(fast, ref) < 1e-9 else "\nPARITY FAIL")
