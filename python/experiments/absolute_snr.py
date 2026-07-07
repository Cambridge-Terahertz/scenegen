"""Absolute-SNR surface coverage — noise-floor anchored (aspect-free) + area-weighted.

Fixes red-team #2 and its follow-ups:
  * grade against a noise floor, not each patch's self-referential potential;
  * anchor that floor to an ASPECT-FREE reference -- a single isotropic point
    scatterer -- NOT the body's brightest specular glint (which scales with the
    biggest facet's area) and NOT a flat plate (whose return is bistatic/aspect-
    dependent, as the elevation probe showed);
  * weight coverage by patch AREA (density weights ~ 1/local point density), so
    the % is fraction-of-AREA, not fraction-of-(non-uniform)-points.

Per-patch SNR(dB) = (patch return, dB above a unit point scatterer) + reference
point-target SNR anchor. Coverage@SNR = AREA fraction clearing a detection
threshold, swept over the anchor. A matched noise-only recon confirms the floor
is spatially uniform (Rayleigh). The anchor stays parametric until real system
numbers (TX power, NF, bandwidth, integration) fix the point-target SNR.

Needs a GPU. Output -> RA_OUTPUT_ROOT/experiments/absolute_snr/
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
from bodies import load_real_scene, density_weights  # noqa: E402
from experiments.coverage_vs_range import sample_nearest  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose", required=True)
    p.add_argument("--decimate", type=int, default=4)
    p.add_argument("--standoff", type=float, default=2.0)
    p.add_argument("--voxel", type=float, default=0.01)
    p.add_argument("--dw-voxel", type=float, default=0.005, help="area-weight cell [m]")
    p.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    p.add_argument("--det-db", default="6,10,13", help="detection SNR thresholds [dB]")
    p.add_argument("--nominal-snr", type=float, default=-15.0,
                   help="reference point-target SNR for the body map [dB]")
    args = p.parse_args()

    out = output_root() / "experiments" / "absolute_snr"
    out.mkdir(parents=True, exist_ok=True)

    q, s = load_real_scene(args.pose, assets=[], zbuffer=True, density_weight=True,
                           standoff=args.standoff)
    q, s = q[::args.decimate].copy(), s[::args.decimate].copy()
    w = density_weights(q, args.dw_voxel)                # per-point AREA weight (mean 1)
    print(f"{len(q):,} visible-front patches (decimate {args.decimate}); area-weighted")

    geom = rf.Scene.demo(ntarg=1, nrx=16_900, nf=128)
    f = geom.k * C0 / (2 * np.pi)
    scene = rf.Scene(q=q, s=s, tx=geom.tx, rx=geom.rx, k=geom.k)

    sig = rf.forward(scene, precision=args.precision)
    grid = build_grid(q, voxel=args.voxel)
    img = np.abs(backproject(sig, tx=scene.tx, rx=scene.rx, f=f, grid=grid, precision=args.precision))
    if rf._HAS_CUPY:
        rf.cp.cuda.Stream.null.synchronize()
    surf = sample_nearest(img, grid, q)

    # aspect-free reference: a single UNIT isotropic point scatterer at the body centre
    c = q.mean(0)
    rscene = rf.Scene(q=c[None, :], s=np.array([1.0]), tx=geom.tx, rx=geom.rx, k=geom.k)
    rsig = rf.forward(rscene, precision=args.precision)
    rgrid = build_grid(np.array([c - 0.12, c + 0.12]), voxel=args.voxel)
    rimg = np.abs(backproject(rsig, tx=geom.tx, rx=geom.rx, f=f, grid=rgrid, precision=args.precision))
    if rf._HAS_CUPY:
        rf.cp.cuda.Stream.null.synchronize()
    ref_peak = float(rimg.max())
    reldb = 20 * np.log10(surf / ref_peak + 1e-30)       # patch dB ABOVE a unit point scatterer

    # matched noise-only recon -> floor spatial-uniformity check (level-independent)
    gpu = rf._HAS_CUPY and type(sig).__module__.startswith("cupy")
    xp = rf.cp if gpu else np
    noise = ((xp.random.standard_normal(sig.shape) + 1j * xp.random.standard_normal(sig.shape))
             .astype(sig.dtype) / np.sqrt(2))
    nimg = np.abs(backproject(noise, tx=scene.tx, rx=scene.rx, f=f, grid=grid, precision=args.precision))
    if rf._HAS_CUPY:
        rf.cp.cuda.Stream.null.synchronize()
    floor_cv = float(nimg.std() / (nimg.mean() + 1e-30))
    print(f"reldb (body vs unit point scatterer): {reldb.min():.0f}..{reldb.max():.0f} dB ; "
          f"noise-floor spatial CoV {floor_cv:.2f} (Rayleigh~0.52)")

    dets = [float(x) for x in args.det_db.split(",")]
    anchors = np.arange(-45.0, 5.1, 2.5)                 # reference point-target SNR [dB]
    W = float(w.sum())

    lines = [f"Absolute-SNR coverage (AREA-weighted, aspect-free anchor) -- {len(q)} patches",
             f"reldb range {reldb.min():.0f}..{reldb.max():.0f} dB ; floor CoV {floor_cv:.2f}",
             "ptSNR  " + "  ".join(f">{d:.0f}dB" for d in dets)]
    covmat = {d: [] for d in dets}       # AREA-weighted
    covpt = {d: [] for d in dets}        # point-counted (for comparison)
    lines[-1] = "ptSNR  " + "  ".join(f">{d:.0f}dB(area|pt)" for d in dets)
    for a in anchors:
        snr = reldb + a
        cell = []
        for d in dets:
            va = 100.0 * (w * (snr >= d)).sum() / W          # AREA fraction
            vp = 100.0 * (snr >= d).mean()                   # fraction of points
            covmat[d].append(va); covpt[d].append(vp)
            cell.append(f"{va:4.0f}|{vp:4.0f}")
        lines.append(f"{a:6.1f}  " + "  ".join(cell))
    txt = "\n".join(lines)
    (out / "absolute_snr.txt").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt)

    # plot 1: area-weighted coverage vs reference point-target SNR
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=130)
    for d in dets:
        ln, = ax.plot(anchors, covmat[d], "o-", lw=2, label=f"AREA-weighted, >{d:.0f} dB")
        ax.plot(anchors, covpt[d], "--", lw=1.3, color=ln.get_color(), alpha=0.6)
    ax.plot([], [], "k--", lw=1.3, label="dashed = point-counted (unweighted)")
    ax.set_xlabel("reference point-target SNR (dB)  [system sensitivity to a unit isotropic scatterer]")
    ax.set_ylabel("coverage (% of body area detectable)")
    ax.set_title("Absolute-SNR coverage — area-weighted vs point-counted\n"
                 "aspect-free point-target anchor")
    ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "coverage_vs_snr.png"); plt.close(fig)

    # plot 2: body map coloured by absolute SNR at the nominal anchor
    snr_nom = reldb + args.nominal_snr
    fig, ax = plt.subplots(figsize=(4, 7), dpi=120)
    sc = ax.scatter(q[:, 0], q[:, 2], c=snr_nom, s=2, cmap="viridis",
                    vmin=0, vmax=max(30.0, float(np.percentile(snr_nom, 99))), linewidths=0)
    fig.colorbar(sc, ax=ax, label="absolute SNR (dB)")
    d10 = 100.0 * (w * (snr_nom >= 10)).sum() / W
    ax.set_aspect("equal"); ax.set_xlabel("width x (m)"); ax.set_ylabel("height z (m)")
    ax.set_title(f"Absolute SNR @ point-target SNR = {args.nominal_snr:.0f} dB\n"
                 f"area detectable (>10 dB) = {d10:.0f}%")
    fig.tight_layout(); fig.savefig(out / "snr_bodymap.png"); plt.close(fig)

    print(f"\nDone. -> {out}")


if __name__ == "__main__":
    main()
