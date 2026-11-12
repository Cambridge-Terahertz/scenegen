"""Reflectarray frequency-domain forward model — CuPy GPU kernel + NumPy mirror.

This is the ported, restructured replacement for the live CUDA mex
``cpp/ForwardModelCudaProject/forward_model_mex.cu``. It fixes the two problems
that reference (``forward_model_ref.py``) established:

* the current mex launches ~2.16M kernels (one per output) and recomputes every
  distance — here a single fused kernel computes all ``(freq, rx)`` outputs;
* the current mex is pure float32 → ~1e-3 phase error from multi-metre
  distances — here distances and the phase argument are computed in float64
  (with float64 range reduction), and only the sincos + accumulation are
  float32, giving ~1e-6.

Design (matches ``forward_model_ref.forward_fast`` exactly in float64):

    sig[f, i] = Σ_tx Σ_t  s_t / (r1·r2) · exp(-j k_f (r1 + r2))

* ``r1 = |TX - q_t|`` is precomputed once (Step 2);
* one GPU thread owns one receiver ``i`` and a tile of ``TILE`` frequencies,
  hoisting the frequency-independent ``r2``/amp/``r1+r2`` out of the frequency
  loop → no atomics, no reduction, and no per-frequency distance recompute;
* ``__sincosf`` per frequency (no recursion — the reference showed the recursion
  needs re-anchoring in float32 and ``__sincosf`` is cheap, so we skip it).

The GPU path needs a CuPy environment + CUDA GPU. With NumPy inputs (or no CuPy)
the call falls back to :func:`_forward_numpy`, a faithful mirror of the kernel
arithmetic that runs anywhere and is what we validate on a CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

try:  # GPU is optional; the module is fully usable on CPU.
    import cupy as cp

    _HAS_CUPY = True
except Exception:  # pragma: no cover - depends on environment
    cp = None
    _HAS_CUPY = False

C0 = 299_792_458.0
_TWO_PI = 2.0 * np.pi

# Frequencies processed per thread (P2). A thread computes r2/amp/R once per
# target and reuses it across this many frequencies, cutting redundant distance
# work by (TILE-1)/TILE. 8 keeps the per-thread register arrays small (2*TILE
# float accumulators + TILE wavenumbers); nf is typically a multiple of it.
_TILE = 8

Backend = Literal["auto", "gpu", "cpu"]

__all__ = ["Scene", "forward", "C0"]


# ---------------------------------------------------------------------------
# CUDA kernel
# ---------------------------------------------------------------------------
# Templated on REAL_T: fp64 keeps full accuracy (~1e-6) but is slow on consumer
# GPUs (1/32 fp64 rate); fp32 is ~10-30x faster but carries the ~1e-3 phase error
# from multi-metre distances (see forward_model_ref.py). __sincosf is always fp32.
_KERNEL_TEMPLATE = r"""
extern "C" __global__
void forward_fused(
    const REAL_T* __restrict__ qx, const REAL_T* __restrict__ qy,
    const REAL_T* __restrict__ qz, const float*  __restrict__ s,
    const REAL_T* __restrict__ rxx, const REAL_T* __restrict__ rxy,
    const REAL_T* __restrict__ rxz,
    const REAL_T* __restrict__ r1all,   // [ntx * ntarg], tx-major: r1all[tx*ntarg + t]
    const REAL_T* __restrict__ kf,      // [nf]
    const int ntarg, const int ntx, const int nrx, const int nf,
    float* __restrict__ sig_re, float* __restrict__ sig_im)
{
    // One thread owns one receiver i and a tile of TILE frequencies. r2, amp
    // and the total path R are frequency-independent, so hoisting them out of
    // the frequency tile removes the (TILE-1)/TILE redundant distance work of
    // the old one-thread-per-(f,i) form.
    const int ntiles = (nf + TILE - 1) / TILE;
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)nrx * ntiles;
    if (tid >= total) return;

    const int i  = (int)(tid % nrx);         // receiver (adjacent tid -> coalesced)
    const int f0 = (int)(tid / nrx) * TILE;  // first frequency of this tile

    const REAL_T rx0 = rxx[i], rx1 = rxy[i], rx2 = rxz[i];
    const REAL_T TWO_PI = 6.283185307179586;
    const REAL_T INV_TWO_PI = 0.15915494309189535;   // compile-time immediate (no reg)

    REAL_T kt[TILE];                         // this tile's wavenumbers
    #pragma unroll
    for (int j = 0; j < TILE; ++j) {
        const int f = f0 + j;
        kt[j] = (f < nf) ? kf[f] : (REAL_T)0; // pad freqs: computed, never written
    }

    float acc_re[TILE], acc_im[TILE];        // float32 accumulation
    #pragma unroll
    for (int j = 0; j < TILE; ++j) { acc_re[j] = 0.0f; acc_im[j] = 0.0f; }

    for (int tx = 0; tx < ntx; ++tx) {
        const REAL_T* r1p = r1all + (long)tx * ntarg;
        for (int t = 0; t < ntarg; ++t) {
            const REAL_T dx = rx0 - qx[t];
            const REAL_T dy = rx1 - qy[t];
            const REAL_T dz = rx2 - qz[t];
            const REAL_T r2 = SQRT_FN(dx * dx + dy * dy + dz * dz);
            const REAL_T r1 = r1p[t];
            const float  amp = (float)((REAL_T)s[t] / (r1 * r2));  // freq-independent
            const REAL_T R   = r1 + r2;

            #pragma unroll
            for (int j = 0; j < TILE; ++j) {
                // phase + range reduction in REAL_T, sincos in fp32
                REAL_T ph = -kt[j] * R;
                ph = ph - TWO_PI * FLOOR_FN(ph * INV_TWO_PI);
                float cs, sn;
                __sincosf((float)ph, &sn, &cs);                  // exp(j ph) = cs + j sn
                acc_re[j] += amp * cs;
                acc_im[j] += amp * sn;
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < TILE; ++j) {
        const int f = f0 + j;
        if (f < nf) {
            const long idx = (long)f * nrx + i;   // output row-major (nf, nrx)
            sig_re[idx] = acc_re[j];
            sig_im[idx] = acc_im[j];
        }
    }
}
"""

# precision -> (cuda real type, sqrt fn, floor fn, numpy dtype)
_PRECISION = {
    "fp64": ("double", "sqrt", "floor", np.float64),
    "fp32": ("float", "sqrtf", "floorf", np.float32),
}
_kernels: dict = {}


def _get_kernel(precision: str):
    if not _HAS_CUPY:
        raise RuntimeError("CuPy is not available; GPU path cannot run.")
    if precision not in _PRECISION:
        raise ValueError(f"precision must be one of {list(_PRECISION)}")
    if precision not in _kernels:
        real, sqrt_fn, floor_fn, _ = _PRECISION[precision]
        src = (_KERNEL_TEMPLATE.replace("REAL_T", real)
               .replace("SQRT_FN", sqrt_fn).replace("FLOOR_FN", floor_fn)
               .replace("TILE", str(_TILE)))
        _kernels[precision] = cp.RawKernel(src, "forward_fused",
                                           options=("--use_fast_math",))
    return _kernels[precision]


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Scene:
    """Inputs to the forward model.

    Attributes
    ----------
    q : (ntarg, 3) scatterer positions [m]
    s : (ntarg,)   real reflectivity per scatterer
    tx : (ntx, 3)  transmitter positions [m] (summed coherently)
    rx : (nrx, 3)  receiver positions [m]
    k : (nf,)      wavenumbers 2*pi*f/c [rad/m], assumed linearly spaced
    """

    q: np.ndarray
    s: np.ndarray
    tx: np.ndarray
    rx: np.ndarray
    k: np.ndarray

    def __post_init__(self) -> None:
        for name, arr, ncol in [("q", self.q, 3), ("tx", self.tx, 3),
                                 ("rx", self.rx, 3)]:
            if arr.ndim != 2 or arr.shape[1] != ncol:
                raise ValueError(f"{name} must be (N, 3), got {arr.shape}")
        if self.s.shape != (self.q.shape[0],):
            raise ValueError(f"s must be (ntarg,)={self.q.shape[0]}, got {self.s.shape}")
        if self.k.ndim != 1:
            raise ValueError(f"k must be 1-D, got {self.k.shape}")

    @property
    def ntarg(self) -> int:
        return self.q.shape[0]

    @property
    def ntx(self) -> int:
        return self.tx.shape[0]

    @property
    def nrx(self) -> int:
        return self.rx.shape[0]

    @property
    def nf(self) -> int:
        return self.k.shape[0]

    @classmethod
    def demo(cls, ntarg: int = 400, nrx: int = 121, nf: int = 128, ntx: int = 1,
             seed: int = 0, f1: float = 275e9, f2: float = 305e9,
             array_size: float = 0.067, standoff: float = 2.0) -> "Scene":
        """A random but physically-plausible scene at the system scale."""
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
        rx = np.column_stack([xx.ravel(), np.zeros(side * side), zz.ravel()])
        tx = np.tile([0.15, 0.0, 0.0], (ntx, 1))
        k = _TWO_PI * np.linspace(f1, f2, nf) / C0
        return cls(q=q, s=s, tx=tx, rx=rx, k=k)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def forward(scene: Scene, backend: Backend = "auto",
            precision: str = "fp64") -> np.ndarray:
    """Compute the forward-model signal ``sig`` of shape ``(nf, nrx)`` complex64.

    Parameters
    ----------
    scene : Scene
    backend : {"auto", "gpu", "cpu"}
        "auto" uses the GPU when CuPy is importable, else the NumPy mirror.
    precision : {"fp64", "fp32"}
        GPU only. fp64 is accurate (~1e-6) but slow on consumer GPUs; fp32 is
        much faster but ~1e-3 (the current-mex accuracy). CPU is always fp64-ish.
    """
    if backend == "gpu" or (backend == "auto" and _HAS_CUPY):
        return _forward_gpu(scene, precision)
    return _forward_numpy(scene)


def _forward_gpu(scene: Scene, precision: str = "fp64") -> np.ndarray:
    """GPU path: single fused RawKernel launch. Requires CuPy + a CUDA GPU."""
    if not _HAS_CUPY:
        raise RuntimeError("CuPy is not available; use backend='cpu'.")
    rdt = _PRECISION[precision][3]
    q = cp.asarray(scene.q, rdt)
    tx = cp.asarray(scene.tx, rdt)
    rx = cp.asarray(scene.rx, rdt)
    s = cp.asarray(scene.s, cp.float32)
    k = cp.asarray(scene.k, rdt)
    ntarg, ntx, nrx, nf = scene.ntarg, scene.ntx, scene.nrx, scene.nf

    # Step 2: precompute r1 = |TX - q| once, tx-major (ntx, ntarg) -> flat
    r1 = cp.linalg.norm(q[None, :, :] - tx[:, None, :], axis=2)   # (ntx, ntarg)
    r1all = cp.ascontiguousarray(r1).ravel().astype(rdt)

    qx = cp.ascontiguousarray(q[:, 0])
    qy = cp.ascontiguousarray(q[:, 1])
    qz = cp.ascontiguousarray(q[:, 2])
    rxx = cp.ascontiguousarray(rx[:, 0])
    rxy = cp.ascontiguousarray(rx[:, 1])
    rxz = cp.ascontiguousarray(rx[:, 2])

    sig_re = cp.zeros(nf * nrx, cp.float32)
    sig_im = cp.zeros(nf * nrx, cp.float32)

    threads = 256
    ntiles = (nf + _TILE - 1) // _TILE          # one thread per (rx, freq-tile)
    blocks = (nrx * ntiles + threads - 1) // threads
    _get_kernel(precision)(
        (blocks,), (threads,),
        (qx, qy, qz, s, rxx, rxy, rxz, r1all, k,
         np.int32(ntarg), np.int32(ntx), np.int32(nrx), np.int32(nf),
         sig_re, sig_im),
    )
    sig = (sig_re + 1j * sig_im).reshape(nf, nrx)
    return cp.asnumpy(sig).astype(np.complex64)


def _forward_numpy(scene: Scene) -> np.ndarray:
    """CPU mirror of the kernel arithmetic (fp64 distance/phase, fp32 sincos/accum).

    Faithful to the GPU kernel so we can validate the algorithm on a CPU; the
    accumulation order differs slightly (vectorised vs sequential) but the
    accuracy is the same to ~1e-6.
    """
    q = scene.q.astype(np.float64)
    tx = scene.tx.astype(np.float64)
    rx = scene.rx.astype(np.float64)
    s = scene.s.astype(np.float64)
    k = scene.k.astype(np.float64)
    nf, nrx = scene.nf, scene.nrx

    sig = np.zeros((nf, nrx), np.complex64)
    for tx_pos in tx:
        r1 = np.linalg.norm(q - tx_pos, axis=1)                  # (ntarg,) fp64, once
        r2 = np.linalg.norm(q[:, None, :] - rx[None, :, :], axis=2)  # (ntarg, nrx) fp64
        rs = r1[:, None] + r2                                    # (ntarg, nrx)
        amp = (s[:, None] / (r1[:, None] * r2)).astype(np.float32)
        for f in range(nf):
            ph = -k[f] * rs                                      # fp64
            ph = ph - _TWO_PI * np.floor(ph / _TWO_PI)           # fp64 range reduction
            phf = ph.astype(np.float32)                          # -> fp32 sincos
            re = (amp * np.cos(phf)).sum(axis=0)
            im = (amp * np.sin(phf)).sum(axis=0)
            sig[f] += (re + 1j * im).astype(np.complex64)
    return sig


# ---------------------------------------------------------------------------
# Tests (pytest-discoverable; CPU only)
# ---------------------------------------------------------------------------
def _reference(scene: Scene) -> np.ndarray:
    """float64 ground truth (independent brute force)."""
    from forward_model_ref import forward_brute
    return forward_brute(scene.q, scene.tx, scene.rx, scene.s, scene.k)


def _relerr(a, b) -> float:
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def test_numpy_mirror_matches_reference():
    scene = Scene.demo(ntarg=150, nrx=49, nf=64)
    assert _relerr(_forward_numpy(scene), _reference(scene)) < 5e-6


def test_numpy_mirror_multitx():
    scene = Scene.demo(ntarg=120, nrx=36, nf=32, ntx=3, seed=3)
    assert _relerr(_forward_numpy(scene), _reference(scene)) < 5e-6


def test_mirror_beats_current_kernel():
    # the restructured (fp64 distance) mirror is far more accurate than all-fp32
    from forward_model_ref import forward_brute
    scene = Scene.demo(ntarg=150, nrx=49, nf=64)
    ref = forward_brute(scene.q, scene.tx, scene.rx, scene.s, scene.k)
    current = forward_brute(scene.q, scene.tx, scene.rx, scene.s, scene.k, phase32=True)
    assert _relerr(_forward_numpy(scene), ref) < _relerr(current, ref) / 20


def test_scene_validation():
    import pytest
    with pytest.raises(ValueError):
        Scene(q=np.zeros((4, 2)), s=np.zeros(4), tx=np.zeros((1, 3)),
              rx=np.zeros((9, 3)), k=np.zeros(8))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scene = Scene.demo()
    print(f"scene: ntarg={scene.ntarg} nrx={scene.nrx} nf={scene.nf} ntx={scene.ntx}")
    print(f"CuPy available: {_HAS_CUPY}  ->  backend = {'gpu' if _HAS_CUPY else 'cpu (mirror)'}")

    ref = _reference(scene)
    mirror = _forward_numpy(scene)
    from forward_model_ref import forward_brute
    current = forward_brute(scene.q, scene.tx, scene.rx, scene.s, scene.k, phase32=True)

    print("\nvalidation (CPU, no GPU needed):")
    print(f"  fused-kernel mirror vs float64 truth : {_relerr(mirror, ref):.2e}")
    print(f"  current all-fp32 kernel vs truth     : {_relerr(current, ref):.2e}")
    print(f"  -> restructured kernel is ~{_relerr(current, ref) / _relerr(mirror, ref):.0f}x more accurate")

    if _HAS_CUPY:
        gpu = _forward_gpu(scene)
        print(f"\n  GPU kernel vs mirror : {_relerr(gpu, mirror):.2e}")
    else:
        print("\n  (run on a CuPy + CUDA GPU host to execute and time the RawKernel)")
