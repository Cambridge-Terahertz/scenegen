"""Multi-sensor absolute-SNR trade — ONE common noise floor, NO range-normalization.

Unlike multi_sensor_trade (which x r1*r2-normalizes and scores vs each patch's own
potential), this uses RAW recon returns against a single common receiver floor, so
the shallow-cant sensors are correctly penalized for imaging farther (R=1/sin theta)
-- both the 1/(r1 r2) amplitude AND the coarser lambda*R/D focusing at range.

Reference: a single UNIT point scatterer at a FIXED 2 m (aspect-free, fixed
position). SNR_patch(dB) = 20*log10(union_raw / ref_peak) + anchor, where anchor is
the reference point-target SNR (system sensitivity), swept. Area-weighted coverage
per cant angle -> does 20 deg still win once range is counted?

Needs a GPU. Output -> RA_OUTPUT_ROOT/experiments/multi_sensor_snr/
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
import raforward as rf                # noqa: E402
from recon import backproject, C0    # noqa: E402
from run_model import build_grid      # noqa: E402
from results import output_root       # noqa: E402
from bodies import load_real_scene, density_weights, zbuffer_cull  # noqa: E402
from experiments.coverage_vs_range import sample_nearest  # noqa: E402
from experiments.multi_sensor_trade import sensor_frame   # noqa: E402


def image_raw(q_all, s_all, geom, f, args):
    """Occlude -> forward -> recon -> sample RAW |recon| (NO range normalization)."""
    keep = zbuffer_cull(q_all, 0.003)
    scene = rf.Scene(q=q_all[keep], s=s_all[keep], tx=geom.tx, rx=geom.rx, k=geom.k)
    sig = rf.forward(scene, precision=args.precision)
    grid = build_grid(q_all, voxel=args.voxel)
    img = np.abs(backproject(sig, tx=scene.tx, rx=scene.rx, f=f, grid=grid, precision=args.precision))
    if rf._HAS_CUPY:
        rf.cp.cuda.Stream.null.synchronize()
    return sample_nearest(img, grid, q_all)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose", required=True)
    p.add_argument("--decimate", type=int, default=4)
    p.add_argument("--half-spacing", type=float, default=1.0)
    p.add_argument("--angles", default="20,30,45,60")
    p.add_argument("--n-transit", type=int, default=5)
    p.add_argument("--transit-extent", type=float, default=0.7)
    p.add_argument("--ref-standoff", type=float, default=2.0, help="fixed reference range [m]")
    p.add_argument("--voxel", type=float, default=0.01)
    p.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    p.add_argument("--det-db", default="6,10,13")
    args = p.parse_args()

    out = output_root() / "experiments" / "multi_sensor_snr"
    out.mkdir(parents=True, exist_ok=True)

    q0, s0 = load_real_scene(args.pose, assets=[], zbuffer=False, density_weight=True)
    q0, s0 = q0[::args.decimate].copy(), s0[::args.decimate].copy()
    w = density_weights(q0, 0.005)                    # per-point AREA weight
    W = float(w.sum())
    print(f"{len(q0):,} patches (decimate {args.decimate}); area-weighted")

    geom = rf.Scene.demo(ntarg=1, nrx=16_900, nf=128)
    f = geom.k * C0 / (2 * np.pi)

    # common reference: a unit point scatterer at a FIXED range (aspect-free, fixed)
    R = args.ref_standoff
    rq = np.array([[0.0, R, 0.0]])
    rscene = rf.Scene(q=rq, s=np.array([1.0]), tx=geom.tx, rx=geom.rx, k=geom.k)
    rsig = rf.forward(rscene, precision=args.precision)
    rgrid = build_grid(np.array([[-0.12, R - 0.12, -0.12], [0.12, R + 0.12, 0.12]]), voxel=args.voxel)
    rimg = np.abs(backproject(rsig, tx=geom.tx, rx=geom.rx, f=f, grid=rgrid, precision=args.precision))
    if rf._HAS_CUPY:
        rf.cp.cuda.Stream.null.synchronize()
    ref_peak = float(rimg.max())
    print(f"common reference: unit point scatterer at {R:.1f} m -> ref_peak {ref_peak:.3g}")

    # body in world/person frame (faces +Y), centred
    c0 = q0.mean(0)
    body = np.column_stack([q0[:, 0] - c0[0], -(q0[:, 1] - c0[1]), q0[:, 2] - c0[2]])

    dets = [float(x) for x in args.det_db.split(",")]
    anchors = np.arange(-40.0, 5.1, 2.5)              # reference point-target SNR [dB]
    angles = [float(a) for a in args.angles.split(",")]

    cov = {a: {d: [] for d in dets} for a in angles}
    reldb_by_angle = {}
    t0 = time.perf_counter()
    for th_deg in angles:
        th = np.radians(th_deg); sn, cs = np.sin(th), np.cos(th)
        cot = cs / sn
        sensors = []
        for side in (+1, -1):
            for face in (+1, -1):
                sensors.append((np.array([side * args.half_spacing, 0.0, 0.0]),
                                np.array([-side * sn, face * cs, 0.0]), face * cot))
        union_raw = np.zeros(len(q0))
        for p_s, d, Y0 in sensors:
            for dy in np.linspace(-args.transit_extent, args.transit_extent, args.n_transit):
                qw = body + np.array([0.0, Y0 + dy, 0.0])
                union_raw = np.maximum(union_raw, image_raw(sensor_frame(qw, p_s, d), s0, geom, f, args))
        reldb = 20 * np.log10(union_raw / ref_peak + 1e-30)      # raw vs fixed 2 m point target
        reldb_by_angle[th_deg] = reldb
        for a in anchors:
            snr = reldb + a
            for d in dets:
                cov[th_deg][d].append(100.0 * (w * (snr >= d)).sum() / W)
        print(f"theta {th_deg:.0f}: reldb {reldb.min():.0f}..{reldb.max():.0f} dB  "
              f"{time.perf_counter()-t0:.0f}s")

    # plot: area coverage @10 dB vs anchor, one line per angle
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), dpi=130)
    dmid = dets[len(dets) // 2]
    for th_deg in angles:
        a1.plot(anchors, cov[th_deg][dmid], "o-", lw=2, label=f"cant {th_deg:.0f}°")
    a1.set_xlabel("reference point-target SNR (dB, at fixed 2 m)")
    a1.set_ylabel(f"AREA coverage (% detectable > {dmid:.0f} dB)")
    a1.set_title("Common-floor absolute-SNR — coverage vs system SNR\n(raw returns; shallow cant penalised for range)")
    a1.set_ylim(0, 100); a1.grid(alpha=.3); a1.legend()
    # reldb-max (brightest achievable) per angle = the range/aspect advantage
    a2.plot(angles, [reldb_by_angle[a].max() for a in angles], "s-", label="brightest patch (dB vs 2 m point)")
    a2.plot(angles, [np.median(reldb_by_angle[a]) for a in angles], "o-", label="median patch")
    a2.set_xlabel("cant angle (deg)"); a2.set_ylabel("raw return vs 2 m point target (dB)")
    a2.set_title("Raw return level vs cant (range penalty visible)")
    a2.set_xticks(angles); a2.grid(alpha=.3); a2.legend()
    fig.tight_layout(); fig.savefig(out / "snr_vs_cant.png"); plt.close(fig)

    lines = [f"Multi-sensor common-floor absolute-SNR -- decimate {args.decimate}, ref @ {R:.0f} m",
             "cant   " + "  ".join(f"cov@{d:.0f}dB(anchor=-20)" for d in dets) + "   reldb_max  reldb_med"]
    ai = int(np.argmin(np.abs(anchors + 20)))          # anchor = -20 dB column
    for th_deg in angles:
        lines.append(f"{th_deg:4.0f}   " +
                     "  ".join(f"{cov[th_deg][d][ai]:14.1f}" for d in dets) +
                     f"   {reldb_by_angle[th_deg].max():8.1f}  {np.median(reldb_by_angle[th_deg]):8.1f}")
    (out / "snr_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nDone. -> {out}")


if __name__ == "__main__":
    main()
