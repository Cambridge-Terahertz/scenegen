"""Surface coverage along a trajectory, scored against a full-aspect potential.

Mirrors the reference coverage method (scratch-threat_config_surface_coverage_
measured.py): a patch is "seen" by a view iff that view recovers it to within
SEEN_DB of its OWN best-achievable return -- not the global peak. The
best-achievable return per patch (S_full, the "potential") comes from a dense
full-aspect REFERENCE SCAN: image the body over a fine yaw sweep (the aperture
subtense is ~1.9 deg and specular glints are ~0.6 deg wide, so the step must be
sub-degree -- see experiments/glint_coherence.py) and take each patch's max.

Returns are RANGE-NORMALIZED (multiplied by r1*r2, the forward 1/(r1 r2)
amplitude) so coverage measures aspect capture, not range falloff -- the pass
deliberately changes range while the reference is at a fixed standoff.

Trajectories:
  --mode approach : body head-on, straight toward the sensor 3.0 -> 1.5 m.
  --mode cross    : body path_deg to the z/boresight axis (faces along its path),
                    crossing the boresight (x=0) at --depth (2 m).

Per position: full coherent forward + backprojection, sample range-normalized
|recon| at every body patch, coverage = #(present & recovered>=SEEN_DB of S_full)
/ #present. Total = union over the pass. Needs a GPU (run on bose).
Output -> RA_OUTPUT_ROOT/experiments/coverage_<mode>/
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
from recon import backproject, C0  # noqa: E402
from run_model import build_grid  # noqa: E402
from results import output_root   # noqa: E402
from bodies import load_real_scene, zbuffer_cull  # noqa: E402

SEEN_DB = [-3.0, -6.0, -10.0, -20.0]
PRESENT_DB = -20.0       # patch is "present" if its potential is within this of the peak
SEEN_DB_MAP = -6.0       # threshold annotated on the body maps
MAP_FLOOR = -30.0        # body-map colour floor (dB of potential recovered)


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


def range_factor(pts, tx_pt, rx_ctr):
    """r1*r2 per point -- undoes the forward 1/(r1 r2) so samples are ~reflectivity."""
    r1 = np.linalg.norm(pts - tx_pt, axis=1)
    r2 = np.linalg.norm(pts - rx_ctr, axis=1)
    return r1 * r2


def image_and_sample(q_all, s_all, geom, f, args, tx_pt, rx_ctr):
    """Occlude -> forward (visible) -> recon (grid over the whole body) -> sample
    range-normalised |recon| at EVERY body patch. Returns length-N vector."""
    keep = zbuffer_cull(q_all, 0.003)
    qv, sv = q_all[keep], s_all[keep]
    scene = rf.Scene(q=qv, s=sv, tx=geom.tx, rx=geom.rx, k=geom.k)
    sig = rf.forward(scene, precision=args.precision)
    grid = build_grid(q_all, voxel=args.voxel)              # cover all patches, incl. occluded
    img = backproject(sig, tx=scene.tx, rx=scene.rx, f=f, grid=grid, precision=args.precision)
    if rf._HAS_CUPY:
        rf.cp.cuda.Stream.null.synchronize()
    return sample_nearest(np.abs(img), grid, q_all) * range_factor(q_all, tx_pt, rx_ctr)


def build_trajectory(q0, args):
    """Return (xlabel, invert_x, positions[(label, xval, q_all_N)]). Each q_all_N
    is the FULL body transformed to that position (occlusion handled downstream)."""
    ctr = q0.mean(0)
    if args.mode == "approach":
        depth0 = q0[:, 1].mean()
        pos = []
        for R in np.linspace(3.0, 1.5, args.n_pos):
            q = q0.copy(); q[:, 1] += (R - depth0)
            pos.append((f"{R:.2f} m", float(R), q))
        return "standoff (m)  — toward sensor ->", True, pos

    # cross: yaw to path_deg off boresight (faces along path), slide along the path
    qy = yaw(q0, -args.path_deg, ctr)
    c0 = qy.mean(0)
    base = qy + np.array([-c0[0], args.depth - c0[1], 0.0])    # boresight crossing at t=0
    th = np.radians(args.path_deg)
    d = np.array([np.sin(th), np.cos(th), 0.0])
    pos = []
    for t in np.linspace(-args.extent, args.extent, args.n_pos):
        q = base + t * d
        c = q.mean(0)
        rng, look = float(np.hypot(c[0], c[1])), np.degrees(np.arctan2(c[0], c[1]))
        pos.append((f"t={t:+.2f} (R={rng:.2f}m, look {look:+.0f}deg)", float(t), q))
    return "along-path position t (m)", False, pos


def build_potential(q0, s0, geom, f, args, tx_pt, rx_ctr):
    """Full-aspect reference scan: per-patch max range-normalised return over a fine
    yaw sweep (+/- ref_range deg, ref_step deg). Returns S_full (length-N)."""
    ctr = q0.mean(0)
    yaws = np.arange(-args.ref_range, args.ref_range + 1e-9, args.ref_step)
    S_full = np.zeros(len(q0))
    print(f"reference scan: {len(yaws)} yaws ({-args.ref_range:.0f}..{args.ref_range:.0f} "
          f"deg, step {args.ref_step:.2f})")
    t0 = time.perf_counter()
    for j, psi in enumerate(yaws):
        samp = image_and_sample(yaw(q0, psi, ctr), s0, geom, f, args, tx_pt, rx_ctr)
        S_full = np.maximum(S_full, samp)
        if j % 20 == 0 or j == len(yaws) - 1:
            print(f"  yaw {psi:+5.1f} ({j+1}/{len(yaws)})  {time.perf_counter()-t0:6.0f}s")
    return S_full


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose", required=True)
    p.add_argument("--mode", choices=["approach", "cross"], default="cross")
    p.add_argument("--decimate", type=int, default=8, help="body stride (8 ~ 1.3M pts)")
    p.add_argument("--path-deg", type=float, default=30.0, help="cross: deg off boresight")
    p.add_argument("--depth", type=float, default=2.0, help="cross: crossing range [m]")
    p.add_argument("--extent", type=float, default=1.2, help="cross: +/- along-path [m]")
    p.add_argument("--n-pos", type=int, default=5)
    p.add_argument("--ref-range", type=float, default=60.0, help="reference yaw sweep +/- [deg]")
    p.add_argument("--ref-step", type=float, default=0.75, help="reference yaw step [deg]")
    p.add_argument("--voxel", type=float, default=0.01)
    p.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    args = p.parse_args()

    tag = f"coverage_{args.mode}" + (f"_p{args.path_deg:.0f}" if args.mode == "cross" else "")
    out = output_root() / "experiments" / tag
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading clean body: {args.pose}")
    q_full, s_full = load_real_scene(args.pose, assets=[], zbuffer=False, density_weight=True)
    q0, s0 = q_full[::args.decimate].copy(), s_full[::args.decimate].copy()
    print(f"  {len(q0):,} patches (decimate {args.decimate})")

    geom = rf.Scene.demo(ntarg=1, nrx=16_900, nf=128)
    f = geom.k * C0 / (2 * np.pi)
    tx_pt = np.atleast_2d(geom.tx)[0]
    rx_ctr = geom.rx.mean(0)

    # 1. per-patch potential (full-aspect reference scan) -- cached, since it is
    # independent of the trajectory (depends only on pose/decimate/ref sweep).
    cache = (output_root() / "experiments" / "potential_cache" /
             f"{Path(args.pose).name}_dec{args.decimate}_r{args.ref_range:.0f}_s{args.ref_step:.2f}.npy")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and len(np.load(cache, mmap_mode="r")) == len(q0):
        S_full = np.load(cache)
        print(f"loaded cached potential: {cache.name}")
    else:
        S_full = build_potential(q0, s0, geom, f, args, tx_pt, rx_ctr)
        np.save(cache, S_full)
        print(f"cached potential -> {cache.name}")
    present = S_full >= S_full.max() * 10 ** (PRESENT_DB / 20.0)
    n_present = int(present.sum())
    print(f"present (potential within {PRESENT_DB:.0f} dB of peak): "
          f"{n_present:,}/{len(q0):,} ({100*n_present/len(q0):.0f}%)")

    # 2. the trajectory
    xlabel, invert_x, positions = build_trajectory(q0, args)
    print(f"mode={args.mode}  {len(positions)} positions")

    surf = np.zeros((len(positions), len(q0)))
    table = []
    for k, (label, xval, q_all) in enumerate(positions):
        surf[k] = image_and_sample(q_all, s0, geom, f, args, tx_pt, rx_ctr)
        row = [xval] + [100.0 * (present & (surf[k] >= S_full * 10 ** (db / 20.0))).sum()
                        / max(n_present, 1) for db in SEEN_DB]
        table.append(row)
        print(f"  {label}: " + " ".join(f"{db:.0f}dB:{row[1+j]:.0f}%"
                                         for j, db in enumerate(SEEN_DB)))

    table = np.array(table)
    smax = surf.max(0)                                   # best over the pass, per patch
    union = [100.0 * (present & (smax >= S_full * 10 ** (db / 20.0))).sum()
             / max(n_present, 1) for db in SEEN_DB]

    # recovered-vs-potential in dB, only where present (for the body maps)
    def recov_db(s):
        r = np.full(len(q0), MAP_FLOOR)
        r[present] = np.clip(20 * np.log10(s[present] / S_full[present] + 1e-30), MAP_FLOOR, 0)
        return r

    # ---- plot 1: coverage vs position + total/union ----
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=130)
    for j, db in enumerate(SEEN_DB):
        ln, = ax.plot(table[:, 0], table[:, 1 + j], "o-", lw=2, label=f"seen >= {db:.0f} dB")
        ax.axhline(union[j], ls="--", lw=1, color=ln.get_color(), alpha=0.6)
    ax.plot([], [], "k--", lw=1, label="total (union over pass)")
    if invert_x:
        ax.invert_xaxis()
    ax.set_xlabel(xlabel); ax.set_ylabel("surface coverage  (% of present, vs potential)")
    ax.set_title(f"Coverage vs full-aspect potential — {args.mode}"
                 + (f"  (path {args.path_deg:.0f}°, {args.depth:.1f} m)" if args.mode == "cross" else "")
                 + f"\nref {len(np.arange(-args.ref_range, args.ref_range+1e-9, args.ref_step))} yaws,"
                   f" decimate {args.decimate}")
    ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "coverage_vs_position.png"); plt.close(fig)

    # ---- plot 2: potential + per-position + union body maps (front view) ----
    xz = (q0[:, 0], q0[:, 2])
    pot_db = np.where(present, np.clip(20 * np.log10(S_full / S_full.max() + 1e-30), MAP_FLOOR, 0),
                      np.nan)
    maps = [("POTENTIAL (S_full)", pot_db, None)]
    for (label, xval, _), s in zip(positions, surf):
        maps.append((label, recov_db(s), xval))
    maps.append(("UNION over pass", recov_db(smax), None))
    for tag, dval, xval in maps:
        fig, ax = plt.subplots(figsize=(4, 7), dpi=120)
        sc = ax.scatter(*xz, c=dval, s=2, cmap="inferno", vmin=MAP_FLOOR, vmax=0, linewidths=0)
        fig.colorbar(sc, ax=ax, label="recovered vs potential (dB)" if xval is not None
                     or tag.startswith("UNION") else "potential (dB)")
        ax.set_aspect("equal"); ax.set_xlabel("width x (m)"); ax.set_ylabel("height z (m)")
        if tag.startswith("POTENTIAL"):
            ax.set_title(f"{tag}\npresent = {100*n_present/len(q0):.0f}% of body")
        else:
            cov = 100.0 * (present & (dval >= SEEN_DB_MAP)).sum() / max(n_present, 1)
            ax.set_title(f"{tag}\ncoverage@{SEEN_DB_MAP:.0f}dB = {cov:.0f}%")
        fig.tight_layout()
        name = ("bodymap_potential.png" if tag.startswith("POTENTIAL") else
                "bodymap_union.png" if tag.startswith("UNION") else f"bodymap_{xval:+.2f}.png")
        fig.savefig(out / name); plt.close(fig)

    # ---- plot 3: table ----
    lines = [f"Coverage vs full-aspect potential (% of present surface) -- mode={args.mode}",
             f"present={n_present} patches; ref {args.ref_range:.0f}deg/{args.ref_step:.2f}deg; "
             f"decimate {args.decimate}",
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
