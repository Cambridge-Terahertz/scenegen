"""How fast does a patch's recon return decorrelate as the body yaws?

Sets the yaw step needed for a full-aspect reference scan: image the body at a
fixed standoff over fine yaw steps, sample |recon| at a FIXED set of surface
patches (rotated with the body so we track the same physical facets), and report
(a) the drop of the initially-bright patches vs yaw and (b) the correlation of
the whole per-patch brightness vector vs yaw. The angle where these fall off is
the glint coherence angle -> reference-scan yaw step ~ that angle.

Needs a GPU. Output -> RA_OUTPUT_ROOT/experiments/glint_coherence/
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
from experiments.coverage_vs_range import sample_nearest, yaw  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose", required=True)
    p.add_argument("--decimate", type=int, default=8, help="speed: probe is geometric")
    p.add_argument("--standoff", type=float, default=2.0)
    p.add_argument("--max-yaw", type=float, default=8.0, help="deg")
    p.add_argument("--step", type=float, default=0.5, help="deg")
    p.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    args = p.parse_args()

    out = output_root() / "experiments" / "glint_coherence"
    out.mkdir(parents=True, exist_ok=True)

    q_full, _ = load_real_scene(args.pose, assets=[], zbuffer=False,
                               density_weight=True, standoff=args.standoff)
    q_full = q_full[::args.decimate]
    s_full = np.full(len(q_full), 1.0)
    ctr = q_full.mean(0)
    print(f"{len(q_full):,} body points (decimate {args.decimate})")

    geom = rf.Scene.demo(ntarg=1, nrx=16_900, nf=128)
    f = geom.k * C0 / (2 * np.pi)

    # fixed reference patch set = the head-on visible front surface
    keep0 = zbuffer_cull(q_full, 0.003)
    qref0 = q_full[keep0]                       # these physical facets, tracked across yaw

    yaws = np.arange(0.0, args.max_yaw + 1e-9, args.step)
    surf = []
    for psi in yaws:
        qy = yaw(q_full, psi, ctr)              # rotate whole body
        keep = zbuffer_cull(qy, 0.003)          # visible front at this aspect
        qv, sv = qy[keep], s_full[keep]
        scene = rf.Scene(q=qv, s=sv, tx=geom.tx, rx=geom.rx, k=geom.k)
        sig = rf.forward(scene, precision=args.precision)
        grid = build_grid(qv, voxel=0.01)
        img = backproject(sig, tx=scene.tx, rx=scene.rx, f=f, grid=grid,
                          precision=args.precision)
        if rf._HAS_CUPY:
            rf.cp.cuda.Stream.null.synchronize()
        qref = yaw(qref0, psi, ctr)             # same facets, rotated to this aspect
        surf.append(sample_nearest(np.abs(img), grid, qref))
        print(f"  yaw {psi:4.1f} deg done")

    surf = np.array(surf)                        # (n_yaw, n_ref_patch)
    s0 = surf[0]
    bright = s0 >= np.quantile(s0, 0.90)         # top-10% at head-on

    # (a) mean return of the head-on-bright patches vs yaw (dB rel to yaw 0)
    bright_db = 20 * np.log10(surf[:, bright].mean(1) / (surf[0, bright].mean() + 1e-30))
    # (b) normalised correlation of the per-patch brightness vector vs yaw
    corr = (surf @ s0) / (np.linalg.norm(surf, axis=1) * np.linalg.norm(s0) + 1e-30)

    def crossing(y, val, target):
        below = np.where(val <= target)[0]
        return float(np.interp(target, val[below[0]-1:below[0]+1][::-1],
                               y[below[0]-1:below[0]+1][::-1])) if len(below) and below[0] else np.nan

    a3 = crossing(yaws, bright_db, -3.0)
    a6 = crossing(yaws, bright_db, -6.0)
    chalf = crossing(yaws, corr, 0.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=130)
    ax1.plot(yaws, bright_db, "o-")
    for db, a in [(-3, a3), (-6, a6)]:
        ax1.axhline(db, ls=":", color="0.6")
        if np.isfinite(a):
            ax1.axvline(a, ls="--", color="tab:red", alpha=.5)
            ax1.text(a, db, f"  {a:.1f}°", color="tab:red", fontsize=9)
    ax1.set_xlabel("body yaw (deg)"); ax1.set_ylabel("bright-patch return (dB rel. head-on)")
    ax1.set_title("Glint magnitude decorrelation"); ax1.grid(alpha=.3)

    ax2.plot(yaws, corr, "o-", color="tab:green")
    ax2.axhline(0.5, ls=":", color="0.6")
    if np.isfinite(chalf):
        ax2.axvline(chalf, ls="--", color="tab:red", alpha=.5)
        ax2.text(chalf, 0.5, f"  {chalf:.1f}°", color="tab:red", fontsize=9)
    ax2.set_xlabel("body yaw (deg)"); ax2.set_ylabel("per-patch brightness correlation")
    ax2.set_title("Glint pattern decorrelation"); ax2.set_ylim(0, 1.02); ax2.grid(alpha=.3)
    fig.suptitle(f"Glint coherence vs yaw  (aperture D/R subtense ~1.9 deg)")
    fig.tight_layout(); fig.savefig(out / "glint_coherence.png"); plt.close(fig)

    print(f"\nbright-patch -3 dB at  {a3:.2f} deg ; -6 dB at {a6:.2f} deg")
    print(f"pattern corr=0.5 at    {chalf:.2f} deg")
    print(f"-> suggested reference-scan yaw step ~ {min(a3, chalf):.2f} deg")
    print(f"   (over +/-60 deg that is ~{int(120/max(min(a3, chalf),0.1))} yaws)")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
