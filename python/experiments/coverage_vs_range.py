"""Surface coverage vs standoff (B: recon-based) — conveyor 3.0 -> 1.5 m.

Mirrors the reflect_array_imaging *measured* coverage calc, but in simulation:
run the full coherent forward model + backprojection at each standoff, then ask
what fraction of the body surface is actually RECOVERED in the reconstruction
above a SEEN threshold. Because the forward model is coherent and the surface is
densely sampled (<= ~lambda/3), specular selectivity emerges from interference --
a patch only lights up when its specular lobe reaches the aperture -- so coverage
grows as the body nears the sensor (the aperture subtends more, more lobes land
on it, and resolution improves).

Method (per standoff R):
  1. translate the (clean, no-gun) body to depth R, forward -> sig, recon -> img.
  2. sample |img| at each front-surface point (nearest voxel) = recovered amplitude.
  3. seen(p) = recovered(p) >= peak * 10^(SEEN_DB/20);  coverage = #seen / #surface.

Needs a GPU (run on bose). Outputs -> RA_OUTPUT_ROOT/experiments/coverage_vs_range/
  coverage_vs_range.png   coverage % vs standoff, one line per SEEN_DB
  bodymap_<R>m.png        front view, surface coloured by recovered dB
  coverage_table.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import raforward as rf            # noqa: E402
from recon import backproject, C0  # noqa: E402
from run_model import build_grid  # noqa: E402
from results import output_root   # noqa: E402
from bodies import load_real_scene  # noqa: E402

STANDOFFS = [3.0, 2.625, 2.25, 1.875, 1.5]      # m (toward the sensor)
SEEN_DB = [-3.0, -6.0, -10.0, -20.0]            # recovered-amplitude thresholds
SEEN_DB_MAP = -10.0


def sample_nearest(img_abs, grid, pts):
    """Nearest-voxel |img| at each 3-D point (pure numpy, no scipy)."""
    out = np.zeros(len(pts), np.float32)
    idx = []
    for a, ax in enumerate((grid.xs, grid.ys, grid.zs)):
        i = np.round((pts[:, a] - ax[0]) / (ax[1] - ax[0])).astype(np.int64)
        i = np.clip(i, 0, len(ax) - 1)
        idx.append(i)
    return img_abs[idx[0], idx[1], idx[2]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose", required=True, help="pose folder with body/body.xyz")
    p.add_argument("--assets", nargs="*", default=[], help="clean = none (default)")
    p.add_argument("--voxel", type=float, default=0.01)
    p.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    args = p.parse_args()

    out = output_root() / "experiments" / "coverage_vs_range"
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading clean body: {args.pose}")
    q0, s0 = load_real_scene(args.pose, assets=args.assets, zbuffer=True,
                             density_weight=True)
    depth0 = float(q0[:, 1].mean())
    print(f"  {len(q0):,} front-surface scatterers")

    geom = rf.Scene.demo(ntarg=1, nrx=16_900, nf=128)
    f = geom.k * C0 / (2 * np.pi)

    cover = {}      # (R, db) -> seen mask;   recov[R] -> recovered amplitude (dB) per surf pt
    recov = {}
    table = []
    for R in STANDOFFS:
        q = q0.copy()
        q[:, 1] += (R - depth0)                       # move body to standoff R
        scene = rf.Scene(q=q, s=s0, tx=geom.tx, rx=geom.rx, k=geom.k)
        sig = rf.forward(scene, precision=args.precision)
        grid = build_grid(q, voxel=args.voxel)
        img = backproject(sig, tx=scene.tx, rx=scene.rx, f=f, grid=grid,
                          precision=args.precision)
        if rf._HAS_CUPY:
            rf.cp.cuda.Stream.null.synchronize()

        img_abs = np.abs(img)
        peak = float(img_abs.max())
        surf = sample_nearest(img_abs, grid, q)        # recovered amp at each patch
        recov[R] = 20 * np.log10(surf / (peak + 1e-30) + 1e-30)
        row = [R]
        for db in SEEN_DB:
            seen = surf >= peak * 10 ** (db / 20.0)
            cover[(R, db)] = seen
            row.append(100.0 * seen.mean())
        table.append(row)
        print(f"  R={R:.2f} m  coverage " +
              " ".join(f"{db:.0f}dB:{row[1+j]:.0f}%" for j, db in enumerate(SEEN_DB)))

    table = np.array(table)

    # union ("total") over the whole conveyor pass: best recovery per patch
    recov_max = np.max(np.stack([recov[R] for R in STANDOFFS]), axis=0)
    union = [100.0 * (recov_max >= db).mean() for db in SEEN_DB]

    # 1. coverage vs range (solid = per position; dashed = total/union over pass)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=130)
    for j, db in enumerate(SEEN_DB):
        ln, = ax.plot(table[:, 0], table[:, 1 + j], "o-", lw=2, label=f"seen >= {db:.0f} dB")
        ax.axhline(union[j], ls="--", lw=1, color=ln.get_color(), alpha=0.6)
    ax.plot([], [], "k--", lw=1, label="total (union over pass)")
    ax.set_xlabel("standoff (m)   — body moving toward sensor ->")
    ax.invert_xaxis()
    ax.set_ylabel("surface coverage  (% of front-facing surface recovered)")
    ax.set_title("Recon-based surface coverage vs standoff (conveyor)\n"
                 f"clean body, coherent sim, aperture D=6.7 cm")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "coverage_vs_range.png")
    plt.close(fig)

    # 2. per-standoff front-view body maps coloured by recovered dB
    for R in STANDOFFS:
        d = recov[R]
        fig, ax = plt.subplots(figsize=(4, 7), dpi=120)
        sc = ax.scatter(q0[:, 0], q0[:, 2], c=d, s=2, cmap="inferno",
                        vmin=SEEN_DB_MAP, vmax=0, linewidths=0)
        fig.colorbar(sc, ax=ax, label="recovered (dB)")
        ax.set_aspect("equal")
        ax.set_xlabel("width x (m)")
        ax.set_ylabel("height z (m)")
        cov = 100.0 * (d >= SEEN_DB_MAP).mean()
        ax.set_title(f"R={R:.2f} m   coverage@{SEEN_DB_MAP:.0f}dB = {cov:.0f}%")
        fig.tight_layout()
        fig.savefig(out / f"bodymap_{R:.2f}m.png")
        plt.close(fig)

    # 2b. union body map: best recovery per patch over the whole pass
    fig, ax = plt.subplots(figsize=(4, 7), dpi=120)
    sc = ax.scatter(q0[:, 0], q0[:, 2], c=recov_max, s=2, cmap="inferno",
                    vmin=SEEN_DB_MAP, vmax=0, linewidths=0)
    fig.colorbar(sc, ax=ax, label="best recovered (dB)")
    ax.set_aspect("equal")
    ax.set_xlabel("width x (m)")
    ax.set_ylabel("height z (m)")
    ax.set_title(f"UNION over pass (3.0->1.5 m)\n"
                 f"total coverage@{SEEN_DB_MAP:.0f}dB = {100*(recov_max >= SEEN_DB_MAP).mean():.0f}%")
    fig.tight_layout()
    fig.savefig(out / "bodymap_union.png")
    plt.close(fig)

    # 3. table
    lines = ["Recon-based surface coverage (% of front surface recovered)",
             "standoff[m]  " + " ".join(f"{db:.0f}dB" for db in SEEN_DB)]
    for row in table:
        lines.append(f"{row[0]:8.2f}    " + " ".join(f"{v:6.1f}" for v in row[1:]))
    lines.append("-" * len(lines[-1]))
    lines.append(f"{'TOTAL':>8s}    " + " ".join(f"{u:6.1f}" for u in union)
                 + "   (union over the whole pass)")
    (out / "coverage_table.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nDone. -> {out}")


if __name__ == "__main__":
    main()
