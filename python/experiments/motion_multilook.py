"""Motion-centric coverage: multi-look integration + frame-rate requirement.

A walk-through portal images continuously as the body moves, so coverage is the
union/integration over MANY looks, not 5 snapshots. This fine-samples the cross
trajectory and answers two things:

  1. Coverage vs number of frames -- how many looks you need before coverage
     stops climbing (the plateau is set by the glint coherence angle: glints are
     ~0.6 deg wide, so once the per-frame aspect step drops below that, extra
     frames add nothing). Shows the 5-frame runs were badly undersampled.

  2. Frame-rate spec -- the body's aspect to the array rotates at ~v/R; to not
     step over a 0.6 deg glint you need frame spacing <= glint width, i.e.
     fps >= (aspect rate)/(glint width). Falls out as a number vs walk speed.

Scored against the SAME cached per-patch potential as coverage_vs_range.py (run
that first at matching --pose/--decimate so the potential cache exists).
Needs a GPU. Output -> RA_OUTPUT_ROOT/experiments/motion_multilook_p<deg>/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import raforward as rf            # noqa: E402
from recon import C0              # noqa: E402
from results import output_root   # noqa: E402
from bodies import load_real_scene  # noqa: E402
from experiments.coverage_vs_range import (  # noqa: E402
    image_and_sample, yaw, SEEN_DB, PRESENT_DB)

GLINT_DEG = 0.6           # measured -3 dB glint half-width (glint_coherence.py)


def look_angle(x, depth):
    return np.degrees(np.arctan2(x, depth))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose", required=True)
    p.add_argument("--path-deg", type=float, default=30.0)
    p.add_argument("--depth", type=float, default=2.0)
    p.add_argument("--extent", type=float, default=1.2)
    p.add_argument("--n-frames", type=int, default=101, help="fine samples across the pass")
    p.add_argument("--decimate", type=int, default=8)
    p.add_argument("--ref-range", type=float, default=60.0)
    p.add_argument("--ref-step", type=float, default=0.75)
    p.add_argument("--walk-speed", type=float, default=1.4, help="m/s, for fps mapping")
    p.add_argument("--voxel", type=float, default=0.01)
    p.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    args = p.parse_args()

    out = output_root() / "experiments" / f"motion_multilook_p{args.path_deg:.0f}"
    out.mkdir(parents=True, exist_ok=True)

    q_full, s_full = load_real_scene(args.pose, assets=[], zbuffer=False, density_weight=True)
    q0, s0 = q_full[::args.decimate].copy(), s_full[::args.decimate].copy()
    print(f"{len(q0):,} patches (decimate {args.decimate})")

    geom = rf.Scene.demo(ntarg=1, nrx=16_900, nf=128)
    f = geom.k * C0 / (2 * np.pi)
    tx_pt = np.atleast_2d(geom.tx)[0]
    rx_ctr = geom.rx.mean(0)

    cache = (output_root() / "experiments" / "potential_cache" /
             f"{Path(args.pose).name}_dec{args.decimate}_r{args.ref_range:.0f}_s{args.ref_step:.2f}.npy")
    if not (cache.exists() and len(np.load(cache, mmap_mode="r")) == len(q0)):
        sys.exit(f"potential cache missing: {cache}\nrun coverage_vs_range.py first "
                 f"(same --pose/--decimate/--ref-*).")
    S_full = np.load(cache)
    present = S_full >= S_full.max() * 10 ** (PRESENT_DB / 20.0)
    Sp = S_full[present]
    n_present = int(present.sum())
    print(f"loaded potential; present={n_present:,} ({100*n_present/len(q0):.0f}%)")

    # fine trajectory (same cross geometry as coverage_vs_range)
    ctr = q0.mean(0)
    qy = yaw(q0, -args.path_deg, ctr)
    c0 = qy.mean(0)
    base = qy + np.array([-c0[0], args.depth - c0[1], 0.0])
    th = np.radians(args.path_deg)
    d = np.array([np.sin(th), np.cos(th), 0.0])
    ts = np.linspace(-args.extent, args.extent, args.n_frames)

    surf = np.zeros((args.n_frames, n_present), np.float32)
    looks = np.zeros(args.n_frames)
    t0 = time.perf_counter()
    for k, t in enumerate(ts):
        q_all = base + t * d
        surf[k] = image_and_sample(q_all, s0, geom, f, args, tx_pt, rx_ctr)[present]
        c = q_all.mean(0)
        looks[k] = look_angle(c[0], c[1])
        if k % 20 == 0 or k == args.n_frames - 1:
            print(f"  frame {k+1}/{args.n_frames} (t={t:+.2f}, look {looks[k]:+.1f})"
                  f"  {time.perf_counter()-t0:5.0f}s")

    # ---- 1. coverage vs number of frames (evenly-subsampled looks) ----
    def cov(union, db):
        return 100.0 * (union >= Sp * 10 ** (db / 20.0)).mean()

    Ms = sorted({m for m in [1, 2, 3, 5, 9, 15, 21, 31, 41, 61, 81, args.n_frames]
                 if m <= args.n_frames})
    aspect_span = abs(looks[-1] - looks[0])
    cov_vs_m = {db: [] for db in SEEN_DB}
    for M in Ms:
        idx = np.unique(np.linspace(0, args.n_frames - 1, M).astype(int))
        union = surf[idx].max(0)                    # glint-in-any-of-M-frames
        for db in SEEN_DB:
            cov_vs_m[db].append(cov(union, db))
    step_deg = [aspect_span / max(M - 1, 1) for M in Ms]     # per-frame aspect step
    fps = [(M - 1) * args.walk_speed / (2 * args.extent) for M in Ms]  # frames/s at walk speed

    # ---- 2. frame-rate spec: fps >= v*max(dlook/ds)/glint_width ----
    dlook_ds = np.gradient(np.radians(looks), ts)            # rad per metre along path
    max_rate = float(np.max(np.abs(dlook_ds)))               # rad/m
    vs = np.linspace(0.4, 2.2, 40)
    fps_req = vs * max_rate / np.radians(GLINT_DEG)

    # ---- plots ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), dpi=130)
    for db in SEEN_DB:
        a1.plot(Ms, cov_vs_m[db], "o-", lw=2, label=f"{db:.0f} dB")
    a1.axvline(5, ls=":", color="0.5"); a1.text(5, 5, " 5-frame run", fontsize=8, color="0.4")
    knee = 2 * args.extent  # dummy; annotate glint-limited M
    M_glint = aspect_span / GLINT_DEG + 1
    a1.axvline(M_glint, ls="--", color="tab:red", alpha=.6)
    a1.text(M_glint, 60, f" glint-limited\n ~{M_glint:.0f} frames", fontsize=8, color="tab:red")
    a1.set_xlabel(f"frames across the pass  (aspect span {aspect_span:.0f}°)")
    a1.set_ylabel("union coverage (% of present)")
    a1.set_title("Coverage vs number of looks"); a1.grid(alpha=.3); a1.legend(title="seen >=")
    a1b = a1.twiny(); a1b.set_xlim(a1.get_xlim())
    a1b.set_xticks(Ms[::2]); a1b.set_xticklabels([f"{s:.1f}°" for s in step_deg[::2]], fontsize=7)
    a1b.set_xlabel("per-frame aspect step", fontsize=8)

    a2.plot(vs, fps_req, "-", lw=2, color="tab:purple")
    for v in [0.8, 1.4, 2.0]:
        r = v * max_rate / np.radians(GLINT_DEG)
        a2.plot(v, r, "o", color="tab:purple")
        a2.text(v, r, f"  {v} m/s -> {r:.0f} fps", fontsize=8)
    a2.set_xlabel("walk speed (m/s)"); a2.set_ylabel(f"required frame rate (fps, {GLINT_DEG}° glint)")
    a2.set_title(f"Frame-rate spec  (max aspect rate {np.degrees(max_rate):.0f}°/m)")
    a2.grid(alpha=.3)
    fig.suptitle(f"Motion multi-look — path {args.path_deg:.0f}°, {args.n_frames} frames, "
                 f"decimate {args.decimate}")
    fig.tight_layout(); fig.savefig(out / "motion_multilook.png"); plt.close(fig)

    # ---- gap-fill body maps: 5-frame union vs full fine union ----
    xz = (q0[present, 0], q0[present, 2])
    def recov(union):
        return np.clip(20 * np.log10(union / Sp + 1e-30), -30, 0)
    idx5 = np.unique(np.linspace(0, args.n_frames - 1, 5).astype(int))
    for name, u in [("5-frame union", surf[idx5].max(0)), ("full fine union", surf.max(0))]:
        fig, ax = plt.subplots(figsize=(4, 7), dpi=120)
        sc = ax.scatter(*xz, c=recov(u), s=2, cmap="inferno", vmin=-30, vmax=0, linewidths=0)
        fig.colorbar(sc, ax=ax, label="recovered vs potential (dB)")
        ax.set_aspect("equal"); ax.set_xlabel("width x (m)"); ax.set_ylabel("height z (m)")
        c6 = 100.0 * (u >= Sp * 10 ** (-6 / 20.0)).mean()
        ax.set_title(f"{name}\ncoverage@-6dB = {c6:.0f}%")
        fig.tight_layout()
        fig.savefig(out / (f"bodymap_{'5frame' if '5' in name else 'fine'}.png")); plt.close(fig)

    # ---- summary ----
    lines = [f"Motion multi-look -- path {args.path_deg:.0f} deg, {args.n_frames} frames, "
             f"aspect span {aspect_span:.0f} deg, decimate {args.decimate}",
             f"max aspect rate {np.degrees(max_rate):.1f} deg/m; glint {GLINT_DEG} deg",
             f"required fps @ {args.walk_speed} m/s: "
             f"{args.walk_speed*max_rate/np.radians(GLINT_DEG):.0f}",
             "",
             f"{'frames':>7s} {'aspstep':>8s} {'fps':>6s}  " + " ".join(f"{db:.0f}dB" for db in SEEN_DB)]
    for i, M in enumerate(Ms):
        lines.append(f"{M:7d} {step_deg[i]:7.2f}° {fps[i]:6.1f}  "
                     + " ".join(f"{cov_vs_m[db][i]:6.1f}" for db in SEEN_DB))
    (out / "motion_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nDone. -> {out}")


if __name__ == "__main__":
    main()
