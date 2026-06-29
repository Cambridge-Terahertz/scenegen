"""Full end-to-end run: build/load a cloud -> forward -> recon -> save to NAS.

    python run_model.py                          # synthetic body, real array, GPU
    python run_model.py --n-points 1000000       # bigger cloud
    python run_model.py --xyz /path/body.xyz     # real cloud
    python run_model.py --precision fp64          # higher accuracy (slower)

Saves a layered run folder (results.h5 + meta.json + preview/max_projections.png)
under RA_OUTPUT_ROOT (default = the bose NAS mount).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import raforward as rf
from recon import ReconGrid, backproject, C0
from results import save_run, run_dir
from bodies import synthetic_body, load_xyz, load_real_scene


def build_grid(q, *, pad_xy=0.1, pad_depth=0.1, voxel=0.01):
    lo, hi = q.min(0), q.max(0)
    xs = np.arange(lo[0] - pad_xy, hi[0] + pad_xy, voxel)
    ys = np.arange(lo[1] - pad_depth, hi[1] + pad_depth, voxel)   # depth
    zs = np.arange(lo[2] - pad_xy, hi[2] + pad_xy, voxel)
    return ReconGrid(xs=xs, ys=ys, zs=zs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", help="folder with body/ + assets/ (real data)")
    p.add_argument("--decimate", type=int, default=1, help="body stride (e.g. 3)")
    p.add_argument("--assets", nargs="*", default=["phone"],
                   help="placed assets to include (default: phone)")
    p.add_argument("--no-zbuffer", action="store_true",
                   help="keep occluded (back) surface points")
    p.add_argument("--zbuffer-pixel", type=float, default=0.001)
    p.add_argument("--no-density-weight", action="store_true",
                   help="don't area-normalise non-uniform meshing")
    p.add_argument("--xyz", help="single real .xyz cloud")
    p.add_argument("--n-points", type=int, default=200_000)
    p.add_argument("--no-asset", action="store_true")
    p.add_argument("--nrx", type=int, default=16_900, help="130x130 array")
    p.add_argument("--nf", type=int, default=128)
    p.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    p.add_argument("--voxel", type=float, default=0.01)
    p.add_argument("--run-id", default="synthetic_body")
    args = p.parse_args()

    # --- cloud ---
    if args.scene:
        print(f"real scene: {args.scene}  (body 1:{args.decimate}, assets={args.assets})")
        q, s = load_real_scene(args.scene, decimate=args.decimate, assets=args.assets,
                               zbuffer=not args.no_zbuffer, zbuffer_pixel=args.zbuffer_pixel,
                               density_weight=not args.no_density_weight)
        print(f"  total {len(q):,} scatterers")
    elif args.xyz:
        q = load_xyz(args.xyz).astype(np.float64)
        s = np.full(len(q), 0.7)
        print(f"loaded {len(q):,} points from {args.xyz}")
    else:
        q, s = synthetic_body(args.n_points, with_asset=not args.no_asset)
        print(f"synthetic body: {len(q):,} scatterers"
              f"{' + asset' if not args.no_asset else ''}")

    # --- system geometry (real array + chirp) ---
    geom = rf.Scene.demo(ntarg=1, nrx=args.nrx, nf=args.nf)
    scene = rf.Scene(q=q, s=s, tx=geom.tx, rx=geom.rx, k=geom.k)
    f = scene.k * C0 / (2 * np.pi)
    grid = build_grid(q, voxel=args.voxel)
    nvox = int(np.prod(grid.shape))
    print(f"array: nrx={scene.nrx:,}  nf={scene.nf}  |  recon grid {grid.shape} = {nvox:,} voxels")
    print(f"precision: {args.precision}  backend: {'gpu' if rf._HAS_CUPY else 'cpu'}")

    # --- forward ---
    t0 = time.perf_counter()
    sig = rf.forward(scene, precision=args.precision)
    if rf._HAS_CUPY:
        rf.cp.cuda.Stream.null.synchronize()
    t_fwd = time.perf_counter() - t0
    print(f"forward : {t_fwd:7.2f} s  ({scene.ntarg*scene.nrx*scene.nf/t_fwd/1e9:.1f} G-eval/s)")

    # --- recon ---
    t0 = time.perf_counter()
    img = backproject(sig, tx=scene.tx, rx=scene.rx, f=f, grid=grid,
                      npad=4096, precision=args.precision)
    if rf._HAS_CUPY:
        rf.cp.cuda.Stream.null.synchronize()
    t_rec = time.perf_counter() - t0
    print(f"recon   : {t_rec:7.2f} s  ({nvox*scene.nrx/t_rec/1e9:.1f} G-(vox*rx)/s)")

    # --- save ---
    meta = {
        "run_id": args.run_id, "source": args.xyz or "synthetic_body",
        "ntarg": int(scene.ntarg), "nrx": int(scene.nrx), "nf": int(scene.nf),
        "precision": args.precision, "voxel_m": args.voxel,
        "grid_shape": list(grid.shape),
        "timing_s": {"forward": round(t_fwd, 3), "recon": round(t_rec, 3)},
    }
    d = save_run(run_dir(args.run_id), img=img, grid=grid, meta=meta,
                 sig=sig, f=f, tx=scene.tx, rx=scene.rx, targ=scene.q)
    print(f"\nsaved -> {d}")
    print(f"preview: {d}/preview/max_projections.png")


if __name__ == "__main__":
    main()
