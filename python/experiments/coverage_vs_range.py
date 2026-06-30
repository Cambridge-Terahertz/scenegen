"""Surface coverage along a trajectory (recon-based) for one clean body.

Two trajectories (same coverage metric, mirrors reflect_array measured coverage):
  --mode approach : body head-on, moves straight toward the sensor 3.0 -> 1.5 m.
  --mode cross    : body yawed (default 30 deg, oblique 3/4 view) at a fixed
                    depth (default 2 m), crossing laterally through the field of
                    view. Occlusion is recomputed for the rotated body (a
                    different surface faces the sensor at 30 deg).

Per position: run the full coherent forward + backprojection, sample |recon| at
each visible surface patch (nearest voxel), and report the fraction recovered
above a SEEN threshold. Coherent dense-surface sampling reproduces specular
selectivity, so coverage depends on which facets glint toward the aperture.
Also reports TOTAL (union over the whole pass) coverage.

Needs a GPU (run on bose). Outputs -> RA_OUTPUT_ROOT/experiments/coverage_<mode>/
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
from bodies import load_real_scene, zbuffer_cull  # noqa: E402

SEEN_DB = [-3.0, -6.0, -10.0, -20.0]
SEEN_DB_MAP = -10.0


def sample_nearest(img_abs, grid, pts):
    """Nearest-voxel |img| at each 3-D point (pure numpy)."""
    i = [np.clip(np.round((pts[:, a] - ax[0]) / (ax[1] - ax[0])).astype(np.int64),
                 0, len(ax) - 1)
         for a, ax in enumerate((grid.xs, grid.ys, grid.zs))]
    return img_abs[i[0], i[1], i[2]]


def yaw(q, psi_deg, center):
    """Rotate (width, depth) about `center` by psi (yaw about the vertical)."""
    c, s = np.cos(np.radians(psi_deg)), np.sin(np.radians(psi_deg))
    x, y = q[:, 0] - center[0], q[:, 1] - center[1]
    return np.column_stack([x * c - y * s + center[0],
                            x * s + y * c + center[1], q[:, 2]])


def build_trajectory(q_full, s_full, args):
    """Return (xlabel, invert_x, positions[(label, xval, q, s)])."""
    if args.mode == "approach":
        keep = zbuffer_cull(q_full, 0.003)           # visible front (head-on)
        qv, sv = q_full[keep], s_full[keep]
        depth0 = qv[:, 1].mean()
        pos = []
        for R in np.linspace(3.0, 1.5, args.n_pos):
            q = qv.copy(); q[:, 1] += (R - depth0)
            pos.append((f"{R:.2f} m", float(R), q, sv))
        return "standoff (m)  — toward sensor ->", True, pos

    # cross: body oriented path_deg to the z/boresight axis -- it faces ALONG the
    # path (walking direction), so body and path are parallel (both path_deg off
    # boresight, a path_deg aspect). The body keeps that fixed orientation and
    # slides along the straight path, crossing the boresight (x=0) at `depth` (t=0).
    # Re-occlude once at this aspect (a path_deg-rotated surface faces the array).
    ctr = q_full.mean(0)
    qy = yaw(q_full, -args.path_deg, ctr)            # front turned path_deg off boresight
    keep = zbuffer_cull(qy, 0.003)
    qv, sv = qy[keep], s_full[keep]
    c0 = qv.mean(0)
    qv = qv + np.array([-c0[0], args.depth - c0[1], 0.0])   # boresight crossing at t=0
    th = np.radians(args.path_deg)
    d = np.array([np.sin(th), np.cos(th), 0.0])      # path dir, path_deg off boresight
    pos = []
    for t in np.linspace(-args.extent, args.extent, args.n_pos):
        q = qv + t * d
        c = q.mean(0)
        rng, look = float(np.hypot(c[0], c[1])), np.degrees(np.arctan2(c[0], c[1]))
        pos.append((f"t={t:+.2f} (R={rng:.2f}m, look {look:+.0f}deg)", float(t), q, sv))
    return "along-path position t (m)", False, pos


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose", required=True, help="pose folder with body/body.xyz")
    p.add_argument("--mode", choices=["approach", "cross"], default="approach")
    p.add_argument("--path-deg", type=float, default=30.0,
                   help="cross: path angle from boresight (0=head-on, 90=pure lateral)")
    p.add_argument("--depth", type=float, default=2.0, help="cross: midpoint range [m]")
    p.add_argument("--extent", type=float, default=1.2, help="cross: +/- along-path [m]")
    p.add_argument("--n-pos", type=int, default=5)
    p.add_argument("--voxel", type=float, default=0.01)
    p.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    args = p.parse_args()

    out = output_root() / "experiments" / f"coverage_{args.mode}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading clean body: {args.pose}")
    q_full, s_full = load_real_scene(args.pose, assets=[], zbuffer=False,
                                     density_weight=True)
    print(f"  {len(q_full):,} points (full body)")

    xlabel, invert_x, positions = build_trajectory(q_full, s_full, args)
    q_ref = positions[0][2]                           # patch set (fixed across pass)
    print(f"mode={args.mode}  {len(positions)} positions  "
          f"{len(q_ref):,} visible surface patches")

    geom = rf.Scene.demo(ntarg=1, nrx=16_900, nf=128)
    f = geom.k * C0 / (2 * np.pi)

    recov = {}
    table = []
    for label, xval, q, s in positions:
        scene = rf.Scene(q=q, s=s, tx=geom.tx, rx=geom.rx, k=geom.k)
        sig = rf.forward(scene, precision=args.precision)
        grid = build_grid(q, voxel=args.voxel)
        img = backproject(sig, tx=scene.tx, rx=scene.rx, f=f, grid=grid,
                          precision=args.precision)
        if rf._HAS_CUPY:
            rf.cp.cuda.Stream.null.synchronize()
        img_abs = np.abs(img)
        peak = float(img_abs.max())
        surf = sample_nearest(img_abs, grid, q)
        recov[xval] = 20 * np.log10(surf / (peak + 1e-30) + 1e-30)
        row = [xval] + [100.0 * (surf >= peak * 10 ** (db / 20.0)).mean() for db in SEEN_DB]
        table.append(row)
        print(f"  {label}: " +
              " ".join(f"{db:.0f}dB:{row[1+j]:.0f}%" for j, db in enumerate(SEEN_DB)))

    table = np.array(table)
    recov_max = np.max(np.stack([recov[r[0]] for r in table]), axis=0)
    union = [100.0 * (recov_max >= db).mean() for db in SEEN_DB]

    # 1. coverage along the trajectory (solid) + total/union (dashed)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=130)
    for j, db in enumerate(SEEN_DB):
        ln, = ax.plot(table[:, 0], table[:, 1 + j], "o-", lw=2, label=f"seen >= {db:.0f} dB")
        ax.axhline(union[j], ls="--", lw=1, color=ln.get_color(), alpha=0.6)
    ax.plot([], [], "k--", lw=1, label="total (union over pass)")
    if invert_x:
        ax.invert_xaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("surface coverage  (% recovered)")
    ax.set_title(f"Recon-based coverage — {args.mode}"
                 + (f"  (path {args.path_deg:.0f} deg, {args.depth:.1f} m)" if args.mode == "cross" else ""))
    ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "coverage_vs_position.png"); plt.close(fig)

    # 2. per-position + union body maps (front view, coloured by recovered dB)
    for label, xval, q, s in positions + [("UNION", None, q_ref, None)]:
        d = recov_max if xval is None else recov[xval]
        fig, ax = plt.subplots(figsize=(4, 7), dpi=120)
        sc = ax.scatter(q_ref[:, 0], q_ref[:, 2], c=d, s=2, cmap="inferno",
                        vmin=SEEN_DB_MAP, vmax=0, linewidths=0)
        fig.colorbar(sc, ax=ax, label="recovered (dB)")
        ax.set_aspect("equal"); ax.set_xlabel("width x (m)"); ax.set_ylabel("height z (m)")
        cov = 100.0 * (d >= SEEN_DB_MAP).mean()
        tag = "UNION over pass" if xval is None else label
        ax.set_title(f"{tag}\ncoverage@{SEEN_DB_MAP:.0f}dB = {cov:.0f}%")
        fig.tight_layout()
        fig.savefig(out / (f"bodymap_union.png" if xval is None else f"bodymap_{xval:+.2f}.png"))
        plt.close(fig)

    # 3. table
    lines = [f"Recon-based surface coverage (% recovered) -- mode={args.mode}",
             f"{'pos':>10s}  " + " ".join(f"{db:.0f}dB" for db in SEEN_DB)]
    for row in table:
        lines.append(f"{row[0]:10.3f}  " + " ".join(f"{v:6.1f}" for v in row[1:]))
    lines += ["-" * len(lines[-1]),
              f"{'TOTAL':>10s}  " + " ".join(f"{u:6.1f}" for u in union) + "  (union over pass)"]
    (out / "coverage_table.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nDone. -> {out}")


if __name__ == "__main__":
    main()
