"""Multi-sensor portal cant-angle trade.

4 sensors on 2 posts (spacing 2*half_spacing), person walks between them along +Y
facing +Y. Each post has a FORWARD-canted sensor (looks downstream -> images the
BACK as the person recedes) and a BACKWARD-canted sensor (looks upstream -> images
the FRONT as they approach). Cant angle theta is measured from the WALK LINE and
swept over --angles (default 20,30,45).

For each theta: transform the body into each sensor's frame across a short transit
window, sample range-normalised |recon| at every patch (occlusion handled per
sensor, so each naturally sees its own side of the body), union across the 4
sensors, and report coverage vs the cached full-aspect potential + the residual
blind set — vs the best single sensor.

Needs a GPU + the potential cache (run coverage_vs_range.py once at matching
--decimate/--ref-*). Output -> RA_OUTPUT_ROOT/experiments/multi_sensor_trade/
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
from recon import C0                  # noqa: E402
from results import output_root       # noqa: E402
from bodies import load_real_scene    # noqa: E402
from experiments.coverage_vs_range import (  # noqa: E402
    image_and_sample, build_potential, SEEN_DB, PRESENT_DB)


def sensor_frame(q_world, p_s, d):
    """Express world points in a sensor's frame (x'=cross, y'=depth along boresight
    d, z'=height). d is a unit horizontal vector; array sits near y'=0."""
    c = np.array([d[1], -d[0], 0.0])                 # horizontal perp to d
    rel = q_world - p_s
    return np.column_stack([rel @ c, rel @ d, rel[:, 2]])


def local_front(body, cell=0.03):
    """Per-(width,height)-cell front/back split: patches above the LOCAL mean depth
    (person-frame +Y) are front-facing, below are back-facing. View-relevant (which
    side faces the walk direction) and per-limb — unlike a whole-body mean split,
    which mis-assigns limbs that sit off the body's mean depth plane."""
    kx = np.floor(body[:, 0] / cell).astype(np.int64)
    kz = np.floor(body[:, 2] / cell).astype(np.int64)
    _, inv = np.unique(kx * 100003 + kz, return_inverse=True)
    ssum = np.zeros(inv.max() + 1); cnt = np.zeros(inv.max() + 1)
    np.add.at(ssum, inv, body[:, 1]); np.add.at(cnt, inv, 1.0)
    return body[:, 1] > (ssum / cnt)[inv]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose", required=True)
    p.add_argument("--decimate", type=int, default=8)
    p.add_argument("--ref-range", type=float, default=90.0)
    p.add_argument("--ref-step", type=float, default=0.75)
    p.add_argument("--half-spacing", type=float, default=1.0, help="post offset from walk line [m]")
    p.add_argument("--angles", default="20,30,45")
    p.add_argument("--n-transit", type=int, default=7)
    p.add_argument("--transit-extent", type=float, default=0.7, help="+/- along walk per sensor [m]")
    p.add_argument("--voxel", type=float, default=0.01)
    p.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    args = p.parse_args()

    out = output_root() / "experiments" / "multi_sensor_trade"
    out.mkdir(parents=True, exist_ok=True)

    q0, s0 = load_real_scene(args.pose, assets=[], zbuffer=False, density_weight=True)
    q0, s0 = q0[::args.decimate].copy(), s0[::args.decimate].copy()
    print(f"{len(q0):,} patches (decimate {args.decimate})")

    geom = rf.Scene.demo(ntarg=1, nrx=16_900, nf=128)
    f = geom.k * C0 / (2 * np.pi)
    tx_pt, rx_ctr = np.atleast_2d(geom.tx)[0], geom.rx.mean(0)

    cache = (output_root() / "experiments" / "potential_cache" /
             f"{Path(args.pose).name}_dec{args.decimate}_r{args.ref_range:.0f}_s{args.ref_step:.2f}.npy")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and len(np.load(cache, mmap_mode="r")) == len(q0):
        S_full = np.load(cache); print(f"loaded cached potential: {cache.name}")
    else:
        print("building potential ...")
        S_full = build_potential(q0, s0, geom, f, args, tx_pt, rx_ctr)
        np.save(cache, S_full)
    present = S_full >= S_full.max() * 10 ** (PRESENT_DB / 20.0)
    Sp, n_present = S_full[present], int(present.sum())
    print(f"present {n_present:,} ({100*n_present/len(q0):.0f}%)")

    # body in world/person frame: faces +Y (walk dir), centred
    c0 = q0.mean(0)
    body = np.column_stack([q0[:, 0] - c0[0], -(q0[:, 1] - c0[1]), q0[:, 2] - c0[2]])

    def cov(u, db):
        return 100.0 * (present & (u >= S_full * 10 ** (db / 20.0))).sum() / max(n_present, 1)

    angles = [float(a) for a in args.angles.split(",")]
    results, unions, pers = {}, {}, {}
    t0 = time.perf_counter()
    for th_deg in angles:
        th = np.radians(th_deg); sn, cs = np.sin(th), np.cos(th)
        cot = cs / sn
        sensors = []
        for side in (+1, -1):                        # R, L
            for face in (+1, -1):                    # fwd (+), back (-)
                d = np.array([-side * sn, face * cs, 0.0])   # inward + fore/aft
                p_s = np.array([side * args.half_spacing, 0.0, 0.0])
                Y0 = face * cot                       # boresight crosses walk line here
                sensors.append((f"{'R' if side>0 else 'L'}-{'fwd' if face>0 else 'back'}",
                                p_s, d, Y0))
        union_all = np.zeros(len(q0))
        per = {}
        for name, p_s, d, Y0 in sensors:
            u_s = np.zeros(len(q0))
            for dy in np.linspace(-args.transit_extent, args.transit_extent, args.n_transit):
                qw = body + np.array([0.0, Y0 + dy, 0.0])
                surf = image_and_sample(sensor_frame(qw, p_s, d), s0, geom, f, args, tx_pt, rx_ctr)
                u_s = np.maximum(u_s, surf)
            per[name] = u_s
            union_all = np.maximum(union_all, u_s)
        row = {db: cov(union_all, db) for db in SEEN_DB}
        best_single = {db: max(cov(per[n], db) for n in per) for db in SEEN_DB}
        results[th_deg] = {"union": row, "best_single": best_single,
                           "per": {n: {db: cov(per[n], db) for db in SEEN_DB} for n in per}}
        unions[th_deg] = union_all
        pers[th_deg] = per
        print(f"theta {th_deg:.0f}: union " +
              " ".join(f"{db:.0f}dB {row[db]:.0f}%" for db in SEEN_DB) +
              f"  (best-single -10dB {best_single[-10.0]:.0f}%)  {time.perf_counter()-t0:.0f}s")

    # ---- plot: coverage vs cant angle ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), dpi=130)
    for db in SEEN_DB:
        a1.plot(angles, [results[t]["union"][db] for t in angles], "o-", lw=2, label=f"{db:.0f} dB")
    a1.plot(angles, [results[t]["best_single"][-10.0] for t in angles], "s--", color="0.5",
            label="best single sensor (-10 dB)")
    a1.set_xlabel("cant angle from walk line (deg)"); a1.set_ylabel("coverage (% of present)")
    a1.set_title("4-sensor union coverage vs cant"); a1.set_xticks(angles); a1.grid(alpha=.3); a1.legend()
    for db in SEEN_DB:
        a2.plot(angles, [100 - results[t]["union"][db] for t in angles], "o-", lw=2, label=f"{db:.0f} dB")
    a2.set_xlabel("cant angle from walk line (deg)"); a2.set_ylabel("residual BLIND (% of present)")
    a2.set_title("Residual blind vs cant"); a2.set_xticks(angles); a2.grid(alpha=.3); a2.legend()
    fig.suptitle(f"Multi-sensor portal trade — 2 posts, spacing {2*args.half_spacing:.0f} m, "
                 f"decimate {args.decimate}")
    fig.tight_layout(); fig.savefig(out / "trade_vs_angle.png"); plt.close(fig)

    front = local_front(body)                     # view-relevant per-cell front/back split

    # save union + per-sensor arrays so maps re-plot without re-running the GPU work
    np.savez(out / "unions.npz", angles=np.array(angles), present=present,
             S_full=S_full, q0=q0, front=front,
             **{f"u{int(a)}": unions[a] for a in angles},
             **{f"{n}__{int(a)}": pers[a][n] for a in angles for n in pers[a]})

    # ---- (a) FRONT / BACK split union maps per angle (local split -> limbs correct) ----
    for th_deg in angles:
        rec = np.full(len(q0), -30.0)
        rec[present] = np.clip(20 * np.log10(unions[th_deg][present] / Sp + 1e-30), -30, 0)
        fig, axs = plt.subplots(1, 2, figsize=(8, 7), dpi=120)
        for ax, m, tag, xs in [(axs[0], present & front, "FRONT", q0[:, 0]),
                               (axs[1], present & ~front, "BACK", -q0[:, 0])]:
            sc = ax.scatter(xs[m], q0[m, 2], c=rec[m], s=2, cmap="inferno",
                            vmin=-30, vmax=0, linewidths=0)
            c10 = 100.0 * (rec[m] >= -10).sum() / max(m.sum(), 1)
            ax.set_aspect("equal"); ax.set_xlabel("width x (m)")
            ax.set_title(f"{tag}\ncov@-10dB = {c10:.0f}%  ({m.sum():,} pts)")
        axs[0].set_ylabel("height z (m)")
        fig.colorbar(sc, ax=axs, label="recovered vs potential (dB)")
        fig.suptitle(f"4-sensor union, cant {th_deg:.0f}°   (back mirrored L-R = viewed from behind)")
        fig.savefig(out / f"blindmap_{th_deg:.0f}.png"); plt.close(fig)

    # ---- (b) per-sensor "what each sensor sees" maps (each on the side it images) ----
    sees_front = {"R-back": True, "L-back": True, "R-fwd": False, "L-fwd": False}
    for th_deg in angles:
        fig, axs = plt.subplots(2, 2, figsize=(9, 11), dpi=110)
        for ax, name in zip(axs.flat, ["R-back", "L-back", "R-fwd", "L-fwd"]):
            sf = sees_front[name]
            m = present & (front if sf else ~front)
            r = np.clip(20 * np.log10(pers[th_deg][name][m] / S_full[m] + 1e-30), -30, 0)
            xs = q0[m, 0] if sf else -q0[m, 0]
            sc = ax.scatter(xs, q0[m, 2], c=r, s=2, cmap="inferno", vmin=-30, vmax=0, linewidths=0)
            c10 = 100.0 * (r >= -10).sum() / max(m.sum(), 1)
            ax.set_aspect("equal")
            ax.set_title(f"{name}  (sees {'FRONT' if sf else 'BACK'})\ncov@-10dB = {c10:.0f}%")
        fig.colorbar(sc, ax=axs, label="recovered vs potential (dB)")
        fig.suptitle(f"What each sensor sees — cant {th_deg:.0f}°  (back panels mirrored L-R)")
        fig.savefig(out / f"persensor_{th_deg:.0f}.png"); plt.close(fig)

    # ---- summary ----
    lines = [f"Multi-sensor portal trade -- spacing {2*args.half_spacing:.0f} m, decimate {args.decimate}, "
             f"present {n_present}", ""]
    for th_deg in angles:
        r = results[th_deg]
        lines.append(f"cant {th_deg:.0f} deg  union: " +
                     " ".join(f"{db:.0f}dB {r['union'][db]:5.1f}%" for db in SEEN_DB) +
                     f"   | best-single -10dB {r['best_single'][-10.0]:.1f}%")
        for n in r["per"]:
            lines.append(f"    {n:8s} " + " ".join(f"{db:.0f}dB {r['per'][n][db]:5.1f}%" for db in SEEN_DB))
    (out / "trade_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nDone. -> {out}")


if __name__ == "__main__":
    main()
