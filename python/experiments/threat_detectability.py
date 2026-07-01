"""Concealed-threat detectability map.

Sweeps a concealed object (default: the placed gun asset) over a grid of body
locations. For each location it images the multi-look cross pass and scores the
object's return CONTRAST against the local body clutter around it (the honest
detector: no clean reference, the body's own specular glints are the competition).
A location is "detected" if its best-over-pass contrast exceeds --detect-db.

Because the forward model is linear in the scatterers, the body's signal per
frame is computed ONCE and the (cheap) gun signal is added per location -- so the
whole location x frame sweep is fast. Recon is a small local grid around the gun.

Output -> RA_OUTPUT_ROOT/experiments/threat_<asset>/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import raforward as rf                    # noqa: E402
from recon import backproject, ReconGrid, C0  # noqa: E402
from results import output_root           # noqa: E402
from bodies import load_real_scene, load_xyz, place_asset, zbuffer_cull, density_weights  # noqa: E402
from experiments.coverage_vs_range import yaw, sample_nearest  # noqa: E402


def pick_targets(front, n):
    """~n frontmost body points spread over an (x,z) grid."""
    lo, hi = front.min(0), front.max(0)
    g = max(int(np.ceil(np.sqrt(n * 1.6))), 3)
    xs = np.linspace(lo[0], hi[0], g)
    zs = np.linspace(lo[2], hi[2], g)
    picks = []
    for xc in xs:
        for zc in zs:
            m = (np.abs(front[:, 0] - xc) < (xs[1]-xs[0])/2) & \
                (np.abs(front[:, 2] - zc) < (zs[1]-zs[0])/2)
            if m.any():
                sub = front[m]
                picks.append(sub[np.argmin(sub[:, 1])])   # frontmost in cell
    picks = np.array(picks)
    if len(picks) > n:                                     # thin to n, evenly
        picks = picks[np.linspace(0, len(picks)-1, n).astype(int)]
    return picks


def local_grid(center, half=0.15, voxel=0.01):
    ax = lambda c: np.arange(c - half, c + half, voxel)
    return ReconGrid(xs=ax(center[0]), ys=ax(center[1]), zs=ax(center[2]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose", required=True)
    p.add_argument("--asset", default="gun")
    p.add_argument("--decimate", type=int, default=8)
    p.add_argument("--n-locations", type=int, default=30)
    p.add_argument("--asset-decimate", type=int, default=6, help="thin the asset cloud")
    p.add_argument("--frames", type=int, default=17)
    p.add_argument("--path-deg", type=float, default=30.0)
    p.add_argument("--depth", type=float, default=2.0)
    p.add_argument("--extent", type=float, default=1.2)
    p.add_argument("--detect-db", type=float, default=6.0, help="contrast for 'detected'")
    p.add_argument("--voxel", type=float, default=0.01)
    p.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    args = p.parse_args()

    out = output_root() / "experiments" / f"threat_{args.asset}"
    out.mkdir(parents=True, exist_ok=True)

    q_body, s_body = load_real_scene(args.pose, assets=[], zbuffer=False, density_weight=True)
    q_body, s_body = q_body[::args.decimate].copy(), s_body[::args.decimate].copy()

    # raw asset shape (centred) + its reflectivity
    apath = Path(args.pose) / "assets" / f"{args.asset}.xyz"
    ajson = Path(args.pose) / "assets" / "assets.json"
    asset_local = load_xyz(apath).astype(np.float64)[::args.asset_decimate]
    asset_local -= asset_local.mean(0)
    s_asset = json.loads(ajson.read_text())[args.asset]["s"]
    w_asset = density_weights(asset_local, 0.005)      # area-normalise, like the body
    print(f"body {len(q_body):,} pts (dec {args.decimate}); {args.asset} "
          f"{len(asset_local):,} pts s={s_asset} (density-weighted)")

    geom = rf.Scene.demo(ntarg=1, nrx=16_900, nf=128)
    f = geom.k * C0 / (2 * np.pi)

    ctr = q_body.mean(0)
    th = np.radians(args.path_deg)
    d = np.array([np.sin(th), np.cos(th), 0.0])
    ts = np.linspace(-args.extent, args.extent, args.frames)
    yaw0 = -args.path_deg
    # base offset that puts the (yawed) body's boresight crossing at t=0
    qy0 = yaw(q_body, yaw0, ctr)
    base_shift = np.array([-qy0.mean(0)[0], args.depth - qy0.mean(0)[1], 0.0])

    def transform(pts, t):
        return yaw(pts, yaw0, ctr) + base_shift + t * d

    # precompute per-frame body signal (linear forward -> add gun later)
    print(f"precomputing {args.frames} body-frame signals ...")
    sig_body, tbody = [], time.perf_counter()
    for t in ts:
        qb = transform(q_body, t)
        keep = zbuffer_cull(qb, 0.003)
        sc = rf.Scene(q=qb[keep], s=s_body[keep], tx=geom.tx, rx=geom.rx, k=geom.k)
        sig_body.append(rf.forward(sc, precision=args.precision))
    if rf._HAS_CUPY:
        rf.cp.cuda.Stream.null.synchronize()
    print(f"  body signals done ({time.perf_counter()-tbody:.0f}s)")

    front = q_body[zbuffer_cull(q_body, 0.003)]
    targets = pick_targets(front, args.n_locations)
    print(f"{len(targets)} threat locations")

    results = []   # (target_xyz, best_contrast_db)
    t0 = time.perf_counter()
    for i, tgt in enumerate(targets):
        try:
            gun = place_asset(q_body, asset_local, tgt, seed=i)
            s_g = s_asset * w_asset                     # area-fair against the body
            best = -np.inf
            for k, t in enumerate(ts):
                gf = transform(gun, t)
                sc = rf.Scene(q=gf, s=s_g, tx=geom.tx, rx=geom.rx, k=geom.k)
                sig = sig_body[k] + rf.forward(sc, precision=args.precision)
                gc = gf.mean(0)
                grid = local_grid(gc, half=0.15, voxel=args.voxel)
                img = np.abs(backproject(sig, tx=geom.tx, rx=geom.rx, f=f,
                                         grid=grid, precision=args.precision))
                gsig = float(sample_nearest(img, grid, gf).max())      # object peak
                gx, gy, gz = np.meshgrid(grid.xs, grid.ys, grid.zs, indexing="ij")
                r = np.sqrt((gx-gc[0])**2 + (gy-gc[1])**2 + (gz-gc[2])**2)
                clutter = float(np.percentile(img[r > 0.08], 90))      # body surround
                best = max(best, 20*np.log10(gsig/(clutter + 1e-30)))
            results.append((tgt, best))
            print(f"  [{i+1}/{len(targets)}] ({tgt[0]:+.2f},{tgt[2]:+.2f})  "
                  f"contrast {best:+.1f} dB  {'DET' if best>=args.detect_db else 'miss'}")
        except Exception as e:
            print(f"  [{i+1}/{len(targets)}] FAILED: {e}")
    if rf._HAS_CUPY:
        rf.cp.cuda.Stream.null.synchronize()

    tg = np.array([r[0] for r in results])
    cdb = np.array([r[1] for r in results])
    det = cdb >= args.detect_db
    frac = 100.0 * det.mean() if len(det) else 0.0

    fig, ax = plt.subplots(figsize=(4.5, 7.5), dpi=130)
    ax.scatter(front[:, 0], front[:, 2], s=1, c="0.85", linewidths=0)
    sc = ax.scatter(tg[:, 0], tg[:, 2], c=cdb, s=120, cmap="RdYlGn", vmin=0, vmax=15,
                    edgecolors=np.where(det, "k", "0.5"), linewidths=1.2)
    fig.colorbar(sc, ax=ax, label="best-over-pass contrast (dB)")
    ax.set_aspect("equal"); ax.set_xlabel("width x (m)"); ax.set_ylabel("height z (m)")
    ax.set_title(f"{args.asset} detectability — path {args.path_deg:.0f}°, {args.frames} looks\n"
                 f"{frac:.0f}% of {len(det)} locations detected (>= {args.detect_db:.0f} dB)")
    fig.tight_layout(); fig.savefig(out / "detectability_map.png"); plt.close(fig)

    lines = [f"{args.asset} detectability -- path {args.path_deg:.0f} deg, {args.frames} looks, "
             f"decimate {args.decimate}",
             f"detected (>= {args.detect_db:.0f} dB): {det.sum()}/{len(det)} ({frac:.0f}%)", "",
             f"{'x':>7s} {'z':>7s} {'contrast_dB':>12s}  det"]
    for (t, c), dd in zip(results, det):
        lines.append(f"{t[0]:7.2f} {t[2]:7.2f} {c:12.1f}  {'Y' if dd else 'n'}")
    (out / "detectability.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[:3]))
    print(f"total {time.perf_counter()-t0:.0f}s\nDone. -> {out}")


if __name__ == "__main__":
    main()
