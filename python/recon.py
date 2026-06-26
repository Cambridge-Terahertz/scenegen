"""Time-domain backprojection reconstruction — CuPy kernel + NumPy mirror.

Ported from ``bp.m``. The adjoint of ``raforward`` (matched filter): given the
measured ``sig[nf, nrx]``, form per-receiver range profiles by an IFFT over
frequency, then for each voxel sum the (delay-interpolated, carrier-corrected)
contribution from every receiver:

    img(v) = Σ_i  interp( td_i, T_{v,i} ) · exp(-j 2π f1 T_{v,i})
    T_{v,i} = ( |TX - v| + |RX_i - v| ) / c

This is O(nvox · nrx) (the IFFT amortises the frequency sum), which is why the
time-domain form is preferred over a per-voxel DFT when nvox is large.

Precision/backends mirror raforward: fp32 (default, fast, ~1e-3 — current-mex
accuracy) or fp64; GPU via CuPy, CPU via the NumPy mirror.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import raforward as rf  # reuse cp, _HAS_CUPY, _PRECISION, C0
from raforward import C0

__all__ = ["ReconGrid", "backproject"]


# ---------------------------------------------------------------------------
# CUDA kernel: one thread per voxel, delay-and-sum over receivers
# ---------------------------------------------------------------------------
_BP_TEMPLATE = r"""
extern "C" __global__
void bp_kernel(
    const REAL_T* __restrict__ vx, const REAL_T* __restrict__ vy,
    const REAL_T* __restrict__ vz,
    const REAL_T tx0, const REAL_T tx1, const REAL_T tx2,
    const REAL_T* __restrict__ rxx, const REAL_T* __restrict__ rxy,
    const REAL_T* __restrict__ rxz,
    const float* __restrict__ td_re, const float* __restrict__ td_im,  // (nrx, npad) row-major
    const int npad, const int nrx, const int nvox,
    const REAL_T c, const REAL_T f1, const REAL_T dt,
    float* __restrict__ out_re, float* __restrict__ out_im)
{
    const long v = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nvox) return;

    const REAL_T x = vx[v], y = vy[v], z = vz[v];
    REAL_T dx = tx0 - x, dy = tx1 - y, dz = tx2 - z;
    const REAL_T R1 = SQRT_FN(dx * dx + dy * dy + dz * dz);
    const REAL_T TWO_PI = 6.283185307179586;
    const REAL_T P = (REAL_T)npad * dt;           // unambiguous window = 1/df

    float acc_re = 0.0f, acc_im = 0.0f;
    for (int i = 0; i < nrx; ++i) {
        dx = rxx[i] - x; dy = rxy[i] - y; dz = rxz[i] - z;
        const REAL_T R2 = SQRT_FN(dx * dx + dy * dy + dz * dz);
        const REAL_T T = (R1 + R2) / c;

        // range profile is periodic with period P: sample at delay mod P
        const REAL_T tau = T - P * FLOOR_FN(T / P);
        const REAL_T xs = tau / dt;
        const int n = (int)floor(xs);
        const int nn = (n + 1 == npad) ? 0 : (n + 1);   // periodic neighbour
        const float frac = (float)(xs - (REAL_T)n);
        const long base = (long)i * npad;
        const float re = td_re[base + n] * (1.0f - frac) + td_re[base + nn] * frac;
        const float im = td_im[base + n] * (1.0f - frac) + td_im[base + nn] * frac;

        // carrier: exp(+j 2π f1 T)  (conjugate of forward's exp(-j 2π f T))
        REAL_T ph = TWO_PI * f1 * T;
        ph = ph - TWO_PI * FLOOR_FN(ph / TWO_PI);
        float cs, sn;
        __sincosf((float)ph, &sn, &cs);           // exp(j ph) = cs + j sn

        acc_re += re * cs - im * sn;
        acc_im += re * sn + im * cs;
    }
    out_re[v] = acc_re;
    out_im[v] = acc_im;
}
"""

_bp_kernels: dict = {}


def _get_bp_kernel(precision: str):
    if not rf._HAS_CUPY:
        raise RuntimeError("CuPy is not available; GPU path cannot run.")
    if precision not in rf._PRECISION:
        raise ValueError(f"precision must be one of {list(rf._PRECISION)}")
    if precision not in _bp_kernels:
        real, sqrt_fn, floor_fn, _ = rf._PRECISION[precision]
        src = (_BP_TEMPLATE.replace("REAL_T", real)
               .replace("SQRT_FN", sqrt_fn).replace("FLOOR_FN", floor_fn))
        _bp_kernels[precision] = rf.cp.RawKernel(src, "bp_kernel")
    return _bp_kernels[precision]


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReconGrid:
    """Regular voxel grid (ndgrid of xs, ys, zs). y is depth in this geometry."""

    xs: np.ndarray
    ys: np.ndarray
    zs: np.ndarray

    @property
    def shape(self) -> tuple:
        return (self.xs.size, self.ys.size, self.zs.size)

    def voxels(self) -> np.ndarray:
        xx, yy, zz = np.meshgrid(self.xs, self.ys, self.zs, indexing="ij")
        return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _windows(nf: int, freq_window: str) -> np.ndarray:
    if freq_window == "hann":
        return np.hanning(nf)
    if freq_window in ("none", None):
        return np.ones(nf)
    raise ValueError(f"unknown freq_window: {freq_window}")


def _range_profiles(sig: np.ndarray, f: np.ndarray, npad: int,
                    freq_window: str, c: float):
    """IFFT over frequency -> (nrx, npad) range profiles, plus the time axis."""
    nf, nrx = sig.shape
    winf = _windows(nf, freq_window)
    td = np.fft.ifft(sig * winf[:, None], n=npad, axis=0)   # (npad, nrx)
    df = float(f[1] - f[0])
    dt = 1.0 / (df * npad)
    return np.ascontiguousarray(td.T), dt          # (nrx, npad) C-contiguous


def backproject(sig: np.ndarray, *, tx: np.ndarray, rx: np.ndarray, f: np.ndarray,
                grid: ReconGrid, npad: int = 4096, freq_window: str = "hann",
                c: float = C0, backend: str = "auto",
                precision: str = "fp32") -> np.ndarray:
    """Reconstruct an image of shape ``grid.shape`` (complex64).

    sig : (nf, nrx) measured signal
    tx  : (3,) or (1, 3) transmitter position
    rx  : (nrx, 3) receiver positions
    f   : (nf,) frequencies [Hz]
    """
    tx = np.atleast_2d(tx)[0]
    voxels = grid.voxels()
    td, dt = _range_profiles(np.asarray(sig), np.asarray(f), npad, freq_window, c)
    f1 = float(f[0])

    use_gpu = backend == "gpu" or (backend == "auto" and rf._HAS_CUPY)
    if use_gpu:
        img = _backproject_gpu(voxels, tx, rx, td, dt, f1, c, precision)
    else:
        img = _backproject_numpy(voxels, tx, rx, td, dt, f1, c)
    return img.reshape(grid.shape).astype(np.complex64)


def _backproject_numpy(voxels, tx, rx, td, dt, f1, c) -> np.ndarray:
    """CPU mirror: periodic complex interp + carrier, summed over receivers."""
    npad = td.shape[1]
    ti = np.arange(npad) * dt
    period = npad * dt                       # = 1/df
    R1 = np.linalg.norm(voxels - tx, axis=1)
    img = np.zeros(voxels.shape[0], np.complex128)
    for i in range(rx.shape[0]):
        R2 = np.linalg.norm(voxels - rx[i], axis=1)
        T = (R1 + R2) / c
        re = np.interp(T, ti, td[i].real, period=period)   # periodic in 1/df
        im = np.interp(T, ti, td[i].imag, period=period)
        img += (re + 1j * im) * np.exp(1j * 2 * np.pi * f1 * T)
    return img


def _backproject_gpu(voxels, tx, rx, td, dt, f1, c, precision) -> np.ndarray:
    cp = rf.cp
    rdt = rf._PRECISION[precision][3]
    nvox, nrx, npad = voxels.shape[0], rx.shape[0], td.shape[1]

    vx = cp.ascontiguousarray(cp.asarray(voxels[:, 0], rdt))
    vy = cp.ascontiguousarray(cp.asarray(voxels[:, 1], rdt))
    vz = cp.ascontiguousarray(cp.asarray(voxels[:, 2], rdt))
    rxx = cp.ascontiguousarray(cp.asarray(rx[:, 0], rdt))
    rxy = cp.ascontiguousarray(cp.asarray(rx[:, 1], rdt))
    rxz = cp.ascontiguousarray(cp.asarray(rx[:, 2], rdt))
    td_re = cp.ascontiguousarray(cp.asarray(td.real, cp.float32)).ravel()
    td_im = cp.ascontiguousarray(cp.asarray(td.imag, cp.float32)).ravel()
    out_re = cp.zeros(nvox, cp.float32)
    out_im = cp.zeros(nvox, cp.float32)

    rt = lambda a: rdt(a)
    threads = 256
    blocks = (nvox + threads - 1) // threads
    _get_bp_kernel(precision)(
        (blocks,), (threads,),
        (vx, vy, vz, rt(tx[0]), rt(tx[1]), rt(tx[2]),
         rxx, rxy, rxz, td_re, td_im,
         np.int32(npad), np.int32(nrx), np.int32(nvox),
         rt(c), rt(f1), rt(dt), out_re, out_im),
    )
    return cp.asnumpy(out_re + 1j * out_im)


# ---------------------------------------------------------------------------
# End-to-end validation: forward a point scatterer, then reconstruct it.
# ---------------------------------------------------------------------------
def _psf_scene(target_xyz, nrx=400, nf=128):
    """A single point scatterer + the system geometry (reuses raforward.Scene)."""
    demo = rf.Scene.demo(ntarg=1, nrx=nrx, nf=nf)
    q = np.array([target_xyz], float)
    s = np.array([1.0])
    return rf.Scene(q=q, s=s, tx=demo.tx, rx=demo.rx, k=demo.k)


def _peak_location(img, grid):
    idx = np.unravel_index(np.argmax(np.abs(img)), grid.shape)
    return np.array([grid.xs[idx[0]], grid.ys[idx[1]], grid.zs[idx[2]]])


def test_psf_focuses_at_target():
    target = np.array([0.05, 2.0, -0.1])
    scene = _psf_scene(target)
    f = scene.k * C0 / (2 * np.pi)
    sig = rf.forward(scene, backend="cpu")
    grid = ReconGrid(xs=np.linspace(-0.2, 0.2, 41),
                     ys=np.linspace(1.85, 2.15, 31),
                     zs=np.linspace(-0.3, 0.3, 61))
    img = backproject(sig, tx=scene.tx, rx=scene.rx, f=f, grid=grid,
                      npad=2048, backend="cpu")
    peak = _peak_location(img, grid)
    err = np.linalg.norm(peak - target)
    assert err < 0.04, f"PSF peak {peak} vs target {target} (err {err:.3f} m)"


def _relerr(a, b) -> float:
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


if __name__ == "__main__":
    target = np.array([0.05, 2.0, -0.1])
    scene = _psf_scene(target)
    f = scene.k * C0 / (2 * np.pi)
    grid = ReconGrid(xs=np.linspace(-0.2, 0.2, 41),
                     ys=np.linspace(1.85, 2.15, 31),
                     zs=np.linspace(-0.3, 0.3, 61))
    print(f"backend: {'gpu' if rf._HAS_CUPY else 'cpu (mirror)'}")
    print(f"point scatterer at {target}\n")

    if not rf._HAS_CUPY:
        sig = rf.forward(scene, backend="cpu")
        img = backproject(sig, tx=scene.tx, rx=scene.rx, f=f, grid=grid,
                          npad=2048, backend="cpu")
        peak = _peak_location(img, grid)
        print(f"  recon peak {peak}  (err {np.linalg.norm(peak - target) * 1e3:.1f} mm)")
    else:
        # full pipeline in each precision: forward -> recon, then compare images
        imgs = {}
        for prec in ("fp64", "fp32"):
            sig = rf.forward(scene, backend="gpu", precision=prec)
            imgs[prec] = backproject(sig, tx=scene.tx, rx=scene.rx, f=f, grid=grid,
                                     npad=2048, backend="gpu", precision=prec)
            peak = _peak_location(imgs[prec], grid)
            print(f"  {prec}: forward+recon peak {peak}  "
                  f"(err {np.linalg.norm(peak - target) * 1e3:.1f} mm)")
        print(f"\n  fp32 vs fp64 reconstructed-image relerr: "
              f"{_relerr(imgs['fp32'], imgs['fp64']):.2e}")
        print("  -> if this is small, fp32 end-to-end is as good as fp64 (and ~30x faster)")
