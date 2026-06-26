"""Benchmark + validate the CuPy forward-model kernel on a GPU host.

Usage (on the GPU server, inside the venv):

    python bench.py                          # default: decimated-scale scene
    python bench.py --ntarg 4000000          # full ~4M cloud (run_008 scale)
    python bench.py --ntarg 40000 --reps 5   # custom

It (1) checks parity of the GPU kernel against the CPU mirror on a small scene,
then (2) times the GPU kernel at the requested production scale. The wall-clock
here is the number to compare against the current ~50 min mex.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import raforward as rf


def _sync():
    if rf._HAS_CUPY:
        rf.cp.cuda.Stream.null.synchronize()


def parity_check() -> None:
    """GPU vs CPU mirror on a small scene (both should match to ~1e-6)."""
    scene = rf.Scene.demo(ntarg=200, nrx=64, nf=64)
    cpu = rf.forward(scene, backend="cpu")
    if not rf._HAS_CUPY:
        print("parity: CuPy unavailable — skipping GPU parity (CPU mirror only).")
        return
    gpu = rf.forward(scene, backend="gpu")
    rel = float(np.max(np.abs(gpu - cpu)) / np.max(np.abs(cpu)))
    print(f"parity: GPU vs CPU mirror relerr = {rel:.2e}  "
          f"({'OK' if rel < 1e-4 else 'CHECK'})")


def benchmark(ntarg: int, nrx: int, nf: int, reps: int) -> None:
    scene = rf.Scene.demo(ntarg=ntarg, nrx=nrx, nf=nf)
    n_eval = ntarg * nrx * nf
    print(f"\nscene: ntarg={ntarg:,}  nrx={nrx:,}  nf={nf}  "
          f"-> {n_eval:,} target-rx-freq evaluations")

    if not rf._HAS_CUPY:
        print("CuPy unavailable — cannot time the GPU kernel on this host.")
        return

    # warm-up (first launch compiles the RawKernel)
    _ = rf.forward(scene, backend="gpu")
    _sync()

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = rf.forward(scene, backend="gpu")
        _sync()
        times.append(time.perf_counter() - t0)

    best = min(times)
    print(f"GPU forward: best {best * 1e3:8.1f} ms  "
          f"(median {np.median(times) * 1e3:.1f} ms over {reps} reps)")
    print(f"throughput : {n_eval / best / 1e9:.1f} G-eval/s")
    print(f"vs ~50 min current mex: ~{50 * 60 / best:.0f}x faster "
          f"(same ntarg/nrx/nf assumed)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ntarg", type=int, default=40_000, help="scatterers (40k decimated, 4M full)")
    p.add_argument("--nrx", type=int, default=16_900, help="receive elements (130x130)")
    p.add_argument("--nf", type=int, default=128, help="frequencies")
    p.add_argument("--reps", type=int, default=5)
    args = p.parse_args()

    print(f"CuPy available: {rf._HAS_CUPY}")
    if rf._HAS_CUPY:
        dev = rf.cp.cuda.Device()
        props = rf.cp.cuda.runtime.getDeviceProperties(dev.id)
        print(f"GPU: {props['name'].decode()}  (CC {props['major']}.{props['minor']})")

    parity_check()
    benchmark(args.ntarg, args.nrx, args.nf, args.reps)


if __name__ == "__main__":
    main()
