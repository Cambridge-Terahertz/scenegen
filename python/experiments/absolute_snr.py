"""Absolute-SNR surface coverage — grade against a noise floor, not per-patch potential.

Fixes the self-referential-denominator problem (red-team #2): a patch counts only if
its ABSOLUTE return clears a detection threshold above the receiver noise floor — not
if it merely reached its own (possibly weak) best-achievable return.

Method: image the body head-on once, recon the signal and a matched noise-only
realisation (to get the recon noise floor and confirm it is spatially ~uniform).
Per-patch SNR(dB) = (patch return, dB below the peak glint) + system peak-SNR anchor.
Coverage@SNR is swept over a range of system peak-SNR anchors (a link-budget stand-in:
the SNR of the brightest specular glint above noise). As the system SNR drops, patches
whose ABSOLUTE return is weak fall out — which the coverage-vs-potential metric hides.

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
from bodies import load_real_scene    # noqa: E402
from experiments.coverage_vs_range import sample_nearest  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose", required=True)
    p.add_argument("--decimate", type=int, default=4)
    p.add_argument("--standoff", type=float, default=2.0)
    p.add_argument("--voxel", type=float, default=0.01)
    p.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    p.add_argument("--peak-snr", default="15,20,30,40,50", help="system peak-SNR anchors [dB]")
    p.add_argument("--det-db", default="6,10,13", help="detection SNR thresholds [dB]")
    p.add_argument("--nominal-snr", type=float, default=40.0, help="peak SNR for the body map")
    args = p.parse_args()

    out = output_root() / "experiments" / "absolute_snr"
    out.mkdir(parents=True, exist_ok=True)

    q, s = load_real_scene(args.pose, assets=[], zbuffer=True, density_weight=True,
                           standoff=args.standoff)
    q, s = q[::args.decimate].copy(), s[::args.decimate].copy()
    print(f"{len(q):,} visible-front patches (decimate {args.decimate})")

    geom = rf.Scene.demo(ntarg=1, nrx=16_900, nf=128)
    f = geom.k * C0 / (2 * np.pi)
    scene = rf.Scene(q=q, s=s, tx=geom.tx, rx=geom.rx, k=geom.k)

    sig = rf.forward(scene, precision=args.precision)
    grid = build_grid(q, voxel=args.voxel)
    img = np.abs(backproject(sig, tx=scene.tx, rx=scene.rx, f=f, grid=grid, precision=args.precision))
    if rf._HAS_CUPY:
        rf.cp.cuda.Stream.null.synchronize()
    surf = sample_nearest(img, grid, q)
    peak = float(img.max())
    reldb = 20 * np.log10(surf / peak + 1e-30)            # patch dB below the peak glint

    # matched noise-only recon -> floor level + uniformity check
    gpu = rf._HAS_CUPY and type(sig).__module__.startswith("cupy")
    xp = rf.cp if gpu else np
    noise = ((xp.random.standard_normal(sig.shape) + 1j * xp.random.standard_normal(sig.shape))
             .astype(sig.dtype) / np.sqrt(2))
    imgn = np.abs(backproject(noise, tx=scene.tx, rx=scene.rx, f=f, grid=grid, precision=args.precision))
    if rf._HAS_CUPY:
        rf.cp.cuda.Stream.null.synchronize()
    floor = float(np.sqrt((imgn ** 2).mean()))
    floor_cv = float(imgn.std() / (imgn.mean() + 1e-30))  # spatial non-uniformity of the floor
    print(f"recon: peak/unit-noise-floor = {20*np.log10(peak/floor):.1f} dB ; "
          f"noise-floor spatial CoV = {floor_cv:.2f} (should be ~O(0.5), i.e. uniform-ish)")

    anchors = [float(x) for x in args.peak_snr.split(",")]
    dets = [float(x) for x in args.det_db.split(",")]

    lines = [f"Absolute-SNR coverage -- {len(q)} front patches, decimate {args.decimate}",
             f"noise floor spatial CoV {floor_cv:.2f}",
             "peakSNR  " + "  ".join(f">{d:.0f}dB" for d in dets)]
    covmat = {d: [] for d in dets}
    for a in anchors:
        snr = reldb + a                                   # per-patch absolute SNR (dB)
        row = [100.0 * (snr >= d).mean() for d in dets]
        for d, v in zip(dets, row):
            covmat[d].append(v)
        lines.append(f"{a:6.0f}   " + "  ".join(f"{v:5.1f}" for v in row))
    txt = "\n".join(lines)
    (out / "absolute_snr.txt").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt)

    # plot 1: coverage vs system peak SNR
    fig, ax = plt.subplots(figsize=(7, 5), dpi=130)
    for d in dets:
        ax.plot(anchors, covmat[d], "o-", lw=2, label=f"detect > {d:.0f} dB SNR")
    ax.set_xlabel("system peak SNR (dB)  [brightest glint above noise]")
    ax.set_ylabel("surface coverage (% detectable)")
    ax.set_title("Absolute-SNR coverage vs system SNR\n"
                 "patch counts only if its ABSOLUTE return clears noise")
    ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "coverage_vs_snr.png"); plt.close(fig)

    # plot 2: body map coloured by absolute SNR at the nominal anchor
    snr_nom = reldb + args.nominal_snr
    fig, ax = plt.subplots(figsize=(4, 7), dpi=120)
    sc = ax.scatter(q[:, 0], q[:, 2], c=np.clip(snr_nom, 0, None), s=2, cmap="viridis",
                    vmin=0, vmax=args.nominal_snr, linewidths=0)
    fig.colorbar(sc, ax=ax, label="absolute SNR (dB)")
    d10 = 100.0 * (snr_nom >= 10).mean()
    ax.set_aspect("equal"); ax.set_xlabel("width x (m)"); ax.set_ylabel("height z (m)")
    ax.set_title(f"Absolute SNR @ peak-SNR = {args.nominal_snr:.0f} dB\n"
                 f"detectable (>10 dB) = {d10:.0f}%")
    fig.tight_layout(); fig.savefig(out / "snr_bodymap.png"); plt.close(fig)

    print(f"\nDone. -> {out}")


if __name__ == "__main__":
    main()
