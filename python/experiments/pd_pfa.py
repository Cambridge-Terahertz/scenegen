"""#3 Pd/Pfa vs a low-observable threat.

Turns the hard coverage cutoff into real detection statistics. Places a thin
conformal threat sheet on the body, images it (single frontal sensor), injects
NOISE realisations (floor anchored to an aspect-free 2 m point target), runs a
footprint-cell peak detector, and sweeps the threshold -> ROC (Pd vs Pfa). Sweeps
threat reflectivity (= observability). H0 = every threat-free footprint cell on the
body (the body's own glints are the clutter/false alarms); H1 = the threat cell.

Needs a GPU. Output -> RA_OUTPUT_ROOT/experiments/pd_pfa/
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


def recon_complex(q, s, geom, f, args):
    scene = rf.Scene(q=q, s=s, tx=geom.tx, rx=geom.rx, k=geom.k)
    sig = rf.forward(scene, precision=args.precision)
    grid = build_grid(q, voxel=args.voxel)
    img = backproject(sig, tx=scene.tx, rx=scene.rx, f=f, grid=grid, precision=args.precision)
    if rf._HAS_CUPY:
        rf.cp.cuda.Stream.null.synchronize()
    return np.asarray(img), grid


def cell_ids(q, cell):
    kx = np.floor(q[:, 0] / cell).astype(np.int64)
    kz = np.floor(q[:, 2] / cell).astype(np.int64)
    return np.unique(kx * 100003 + kz, return_inverse=True)[1]


def cell_peak(vals, ids, ncell):
    """max of vals over each cell id."""
    out = np.zeros(ncell)
    np.maximum.at(out, ids, vals)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose", required=True)
    p.add_argument("--decimate", type=int, default=6)
    p.add_argument("--standoff", type=float, default=2.0)
    p.add_argument("--threat-refl", default="0.5,1.0,2.0,4.0", help="threat sheet reflectivities")
    p.add_argument("--threat-x", type=float, default=0.20, help="threat centre width [m] (side of torso)")
    p.add_argument("--threat-z", type=float, default=0.10, help="threat centre height [m]")
    p.add_argument("--footprint", type=float, default=0.12, help="threat sheet size [m]")
    p.add_argument("--cell", type=float, default=0.12, help="detector cell size [m]")
    p.add_argument("--anchor", type=float, default=35.0, help="system point-target SNR [dB]")
    p.add_argument("--n-noise", type=int, default=500)
    p.add_argument("--voxel", type=float, default=0.01)
    p.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    args = p.parse_args()

    out = output_root() / "experiments" / "pd_pfa"
    out.mkdir(parents=True, exist_ok=True)

    q, s = load_real_scene(args.pose, assets=[], zbuffer=True, density_weight=True,
                           standoff=args.standoff)
    q, s = q[::args.decimate].copy(), s[::args.decimate].copy()
    print(f"{len(q):,} front patches (decimate {args.decimate})")

    geom = rf.Scene.demo(ntarg=1, nrx=16_900, nf=128)
    f = geom.k * C0 / (2 * np.pi)

    # aspect-free floor anchor: unit point target at fixed 2 m
    c = np.array([0.0, args.standoff, 0.0])
    rimg, rgrid = recon_complex(c[None, :], np.array([1.0]), geom, f, args)
    ref_peak = float(np.abs(rimg).max())
    sigma = ref_peak / 10 ** (args.anchor / 20.0)        # recon noise std (per voxel)
    print(f"ref_peak {ref_peak:.3g}; anchor {args.anchor:.0f} dB -> noise sigma {sigma:.3g}")

    # threat footprint on the body front surface (a box in x,z near the side of the torso)
    fp = ((np.abs(q[:, 0] - args.threat_x) < args.footprint / 2) &
          (np.abs(q[:, 2] - args.threat_z) < args.footprint / 2))
    print(f"threat footprint: {fp.sum():,} patches")

    # clutter (body-only) complex recon sampled at every patch
    img_b, grid_b = recon_complex(q, s, geom, f, args)
    z_body = sample_nearest(img_b, grid_b, q)             # complex per patch

    ids = cell_ids(q, args.cell); ncell = ids.max() + 1
    thr_cell = int(np.bincount(ids[fp]).argmax()) if fp.any() else -1

    rng = np.random.default_rng(0)
    refls = [float(x) for x in args.threat_refl.split(",")]
    pfa_grid = np.logspace(-4, 0, 60)

    def noise(n):
        return sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)

    # H0 clutter cell-peaks (body only): pooled over cells and noise realisations
    H0 = []
    for _ in range(args.n_noise):
        H0.append(cell_peak(np.abs(z_body + noise(len(z_body))), ids, ncell))
    H0 = np.concatenate(H0)                               # all cells x realisations

    results = {}
    for sr in refls:
        # threat = thin conformal sheet just in front of the footprint patches
        sheet = q[fp].copy(); sheet[:, 1] -= 0.003
        qT = np.vstack([q, sheet]); sT = np.concatenate([s, np.full(len(sheet), sr)])
        imgT, gridT = recon_complex(qT, sT, geom, f, args)
        z_thr = sample_nearest(imgT, gridT, q[fp])        # recon at the threat footprint
        H1 = np.array([np.abs(z_thr + noise(len(z_thr))).max() for _ in range(args.n_noise)])
        # ROC
        pd = np.array([(H1 >= t).mean() for t in np.quantile(H0, 1 - pfa_grid)])
        results[sr] = pd
        det = np.quantile(H0, 1 - 0.01)                   # Pfa = 1e-2 threshold
        print(f"s_threat {sr:.1f}: Pd@Pfa=1e-2 = {(H1 >= det).mean():.2f}")

    # ROC plot
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=130)
    for sr in refls:
        ax.semilogx(pfa_grid, results[sr], "-", lw=2, label=f"threat s={sr:.1f}")
    ax.axvline(1e-2, ls=":", color="0.6"); ax.text(1.1e-2, 0.05, "Pfa=1e-2", fontsize=8, color="0.4")
    ax.set_xlabel("Pfa (false-alarm probability)"); ax.set_ylabel("Pd (detection probability)")
    ax.set_title(f"Detection ROC — low-observable threat, single frontal sensor\n"
                 f"system SNR {args.anchor:.0f} dB, threat on torso side; clutter = body glints")
    ax.set_ylim(0, 1.02); ax.grid(alpha=.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "roc.png"); plt.close(fig)

    lines = [f"Pd/Pfa -- {len(q)} patches, system SNR {args.anchor:.0f} dB, "
             f"threat @ ({args.threat_x},{args.threat_z}), footprint {args.footprint*100:.0f} cm",
             "threat_refl   Pd@Pfa=1e-2   Pd@Pfa=1e-3"]
    for sr in refls:
        i2 = np.argmin(np.abs(pfa_grid - 1e-2)); i3 = np.argmin(np.abs(pfa_grid - 1e-3))
        lines.append(f"{sr:9.1f}     {results[sr][i2]:8.2f}     {results[sr][i3]:8.2f}")
    (out / "pd_pfa.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nDone. -> {out}")


if __name__ == "__main__":
    main()
