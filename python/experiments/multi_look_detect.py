"""Multi-look persistence detector — the detection motivation keystone.

A threat is fixed on the body; the body's specular glints are TRANSIENT (each facet
glints only at its own aspect). So averaging each body location over N aspect looks
suppresses the transient glint clutter (~N x) while a persistent threat excess
accumulates. This shows a low-observable threat that a SINGLE look can't detect
(clutter-limited) becomes detectable by the moving/multi-sensor portal.

Images the body over N yaw aspects; per aspect recons the body and the (unit) threat
sheet separately (linear forward -> body + s*sheet, sweep s analytically), adds noise
realisations, and compares SINGLE-look vs MULTI-look-mean detection -> Pd/Pfa ROC.

Needs a GPU. Output -> RA_OUTPUT_ROOT/experiments/multi_look_detect/
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
import raforward as rf                # noqa: E402
from recon import backproject, C0    # noqa: E402
from run_model import build_grid      # noqa: E402
from results import output_root       # noqa: E402
from bodies import load_real_scene, zbuffer_cull  # noqa: E402
from experiments.coverage_vs_range import sample_nearest, yaw  # noqa: E402


def recon_at(q_scene, s_scene, sample_pts, geom, f, args):
    """Recon q_scene and return complex recon sampled at sample_pts."""
    scene = rf.Scene(q=q_scene, s=s_scene, tx=geom.tx, rx=geom.rx, k=geom.k)
    sig = rf.forward(scene, precision=args.precision)
    grid = build_grid(np.vstack([q_scene, sample_pts]), voxel=args.voxel)
    img = backproject(sig, tx=scene.tx, rx=scene.rx, f=f, grid=grid, precision=args.precision)
    if rf._HAS_CUPY:
        rf.cp.cuda.Stream.null.synchronize()
    return sample_nearest(np.asarray(img), grid, sample_pts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose", required=True)
    p.add_argument("--decimate", type=int, default=8)
    p.add_argument("--standoff", type=float, default=2.0)
    p.add_argument("--n-aspect", type=int, default=9)
    p.add_argument("--aspect-span", type=float, default=48.0, help="total yaw sweep [deg]")
    p.add_argument("--threat-refl", default="0.5,1.0,2.0,4.0")
    p.add_argument("--threat-x", type=float, default=0.20)
    p.add_argument("--threat-z", type=float, default=0.10)
    p.add_argument("--footprint", type=float, default=0.12)
    p.add_argument("--cell", type=float, default=0.12)
    p.add_argument("--anchor", type=float, default=35.0)
    p.add_argument("--n-noise", type=int, default=200)
    p.add_argument("--voxel", type=float, default=0.01)
    p.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    args = p.parse_args()

    out = output_root() / "experiments" / "multi_look_detect"
    out.mkdir(parents=True, exist_ok=True)

    q0, s0 = load_real_scene(args.pose, assets=[], zbuffer=False, density_weight=True,
                             standoff=args.standoff)
    q0, s0 = q0[::args.decimate].copy(), s0[::args.decimate].copy()
    ctr = q0.mean(0)
    print(f"{len(q0):,} patches (decimate {args.decimate})")

    geom = rf.Scene.demo(ntarg=1, nrx=16_900, nf=128)
    f = geom.k * C0 / (2 * np.pi)

    # aspect-free floor: lone unit point target at fixed 2 m
    c = np.array([0.0, args.standoff, 0.0])
    rq, rgrid = c[None, :], build_grid(np.array([c - 0.12, c + 0.12]), voxel=args.voxel)
    rimg = np.abs(backproject(rf.forward(rf.Scene(q=rq, s=np.array([1.0]), tx=geom.tx, rx=geom.rx, k=geom.k),
                                         precision=args.precision),
                              tx=geom.tx, rx=geom.rx, f=f, grid=rgrid, precision=args.precision))
    if rf._HAS_CUPY:
        rf.cp.cuda.Stream.null.synchronize()
    ref_peak = float(rimg.max())
    sigma = ref_peak / 10 ** (args.anchor / 20.0)

    # threat sheet in the body frame: footprint patches offset toward the array
    fp = ((np.abs(q0[:, 0] - args.threat_x) < args.footprint / 2) &
          (np.abs(q0[:, 2] - args.threat_z) < args.footprint / 2))
    sheet0 = q0[fp].copy(); sheet0[:, 1] -= 0.003
    print(f"ref_peak {ref_peak:.3g}; sigma {sigma:.3g}; threat footprint {fp.sum()} patches")

    # per-aspect complex recons of body (visible) and unit sheet, sampled at all patches
    aspects = np.linspace(-args.aspect_span / 2, args.aspect_span / 2, args.n_aspect)
    zb = np.zeros((args.n_aspect, len(q0)), np.complex128)
    zs = np.zeros((args.n_aspect, len(q0)), np.complex128)
    for j, psi in enumerate(aspects):
        qy = yaw(q0, psi, ctr)
        keep = zbuffer_cull(qy, 0.003)
        zb[j] = recon_at(qy[keep], s0[keep], qy, geom, f, args)          # body, sampled at all patches
        zs[j] = recon_at(yaw(sheet0, psi, ctr), np.ones(len(sheet0)), qy, geom, f, args)  # unit sheet
        print(f"  aspect {psi:+.0f} deg done")

    ids = np.unique((np.floor(q0[:, 0] / args.cell).astype(np.int64) * 100003 +
                     np.floor(q0[:, 2] / args.cell).astype(np.int64)), return_inverse=True)[1]
    ncell = ids.max() + 1
    thr_cell = int(np.bincount(ids[fp]).argmax())
    mid = args.n_aspect // 2                                             # frontal single look
    rng = np.random.default_rng(0)

    def cpeak(v):
        o = np.zeros(ncell); np.maximum.at(o, ids, v); return o

    refls = [float(x) for x in args.threat_refl.split(",")]
    pfa_grid = np.logspace(-3.5, 0, 50)
    roc = {}
    for sr in refls:
        zt = zb + sr * zs                                               # threat scene, per aspect
        H0s, H1s, H0m, H1m = [], [], [], []
        for _ in range(args.n_noise):
            # single-look (frontal aspect)
            n = sigma * (rng.standard_normal(len(q0)) + 1j * rng.standard_normal(len(q0))) / np.sqrt(2)
            H1s.append(cpeak(np.abs(zt[mid] + n))[thr_cell])
            n = sigma * (rng.standard_normal(len(q0)) + 1j * rng.standard_normal(len(q0))) / np.sqrt(2)
            H0s.extend(cpeak(np.abs(zb[mid] + n)))
            # multi-look mean of magnitude across aspects
            nm = sigma * (rng.standard_normal((args.n_aspect, len(q0))) +
                          1j * rng.standard_normal((args.n_aspect, len(q0)))) / np.sqrt(2)
            H1m.append(cpeak(np.abs(zt + nm).mean(0))[thr_cell])
            nm = sigma * (rng.standard_normal((args.n_aspect, len(q0))) +
                          1j * rng.standard_normal((args.n_aspect, len(q0)))) / np.sqrt(2)
            H0m.extend(cpeak(np.abs(zb + nm).mean(0)))
        H0s, H1s, H0m, H1m = map(np.array, (H0s, H1s, H0m, H1m))

        def roc_of(H0, H1):
            return np.array([(H1 >= t).mean() for t in np.quantile(H0, 1 - pfa_grid)])
        roc[sr] = (roc_of(H0s, H1s), roc_of(H0m, H1m))
        i2 = np.argmin(np.abs(pfa_grid - 1e-2))
        print(f"s={sr:.1f}: Pd@Pfa=1e-2  single {roc[sr][0][i2]:.2f}  multi-look {roc[sr][1][i2]:.2f}")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=130)
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(refls)))
    for sr, cc in zip(refls, cols):
        ax.semilogx(pfa_grid, roc[sr][1], "-", lw=2.3, color=cc, label=f"multi-look, s={sr:.1f}")
        ax.semilogx(pfa_grid, roc[sr][0], "--", lw=1.3, color=cc, alpha=.7)
    ax.plot([], [], "k-", lw=2.3, label="solid = multi-look ({} aspects)".format(args.n_aspect))
    ax.plot([], [], "k--", lw=1.3, label="dashed = single look")
    ax.axvline(1e-2, ls=":", color="0.6")
    ax.set_xlabel("Pfa"); ax.set_ylabel("Pd")
    ax.set_title(f"Persistence detector: single look vs {args.n_aspect}-look mean\n"
                 f"transient glint clutter suppressed by aspect averaging")
    ax.set_ylim(0, 1.02); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "roc_multilook.png"); plt.close(fig)

    lines = [f"Multi-look persistence detector -- {args.n_aspect} aspects, SNR {args.anchor:.0f} dB",
             "threat_refl   Pd@Pfa=1e-2 (single | multi-look)"]
    i2 = np.argmin(np.abs(pfa_grid - 1e-2))
    for sr in refls:
        lines.append(f"{sr:9.1f}      {roc[sr][0][i2]:.2f}  |  {roc[sr][1][i2]:.2f}")
    (out / "multilook.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nDone. -> {out}")


if __name__ == "__main__":
    main()
