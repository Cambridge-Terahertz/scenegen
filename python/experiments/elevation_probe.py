"""Elevation vs azimuth specular-falloff probe.

Validates that the model handles ELEVATION (a 2-D aperture), and measures the
elevation glint width. A dense flat plate sits at the standoff with its normal
facing the array. We sweep the plate's PITCH (elevation tilt, about the width
axis) and its YAW (azimuth, about the vertical) INDEPENDENTLY, and record the
specular peak return vs angle.

- If pitch and yaw fall off similarly, the (square) aperture handles elevation
  correctly and specularity does degrade above/below the sensor.
- If pitch shows little/NO falloff while yaw does, elevation is degenerate (bug).

Needs a GPU. Output -> RA_OUTPUT_ROOT/experiments/elevation_probe/
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


def flat_plate(size=0.12, spacing=0.0005, standoff=2.0):
    """Dense flat plate in the x-z (width-height) plane at y=standoff, normal = -y."""
    n = int(size / spacing)
    a = (np.arange(n) - n / 2) * spacing
    X, Z = np.meshgrid(a, a)
    return np.column_stack([X.ravel(), np.full(X.size, standoff), Z.ravel()])


def rot(q, deg, axis, center):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    if axis == "z":       # yaw about height -> rotates width/depth (azimuth)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:                 # "x": pitch about width -> rotates depth/height (elevation)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return (q - center) @ R.T + center


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=float, default=0.12)
    p.add_argument("--spacing", type=float, default=0.0005)
    p.add_argument("--standoff", type=float, default=2.0)
    p.add_argument("--max-ang", type=float, default=8.0)
    p.add_argument("--step", type=float, default=0.25)
    p.add_argument("--voxel", type=float, default=0.005)
    p.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    args = p.parse_args()

    out = output_root() / "experiments" / "elevation_probe"
    out.mkdir(parents=True, exist_ok=True)

    q = flat_plate(args.size, args.spacing, args.standoff)
    s = np.ones(len(q))
    ctr = q.mean(0)
    print(f"flat plate {args.size*100:.0f} cm, {len(q):,} scatterers "
          f"(spacing {args.spacing*1e3:.2f} mm = lambda/{0.98e-3/args.spacing:.1f})")

    geom = rf.Scene.demo(ntarg=1, nrx=16_900, nf=128)
    f = geom.k * C0 / (2 * np.pi)

    def peak(qq):
        scene = rf.Scene(q=qq, s=s, tx=geom.tx, rx=geom.rx, k=geom.k)
        sig = rf.forward(scene, precision=args.precision)
        grid = build_grid(qq, voxel=args.voxel)
        img = np.abs(backproject(sig, tx=scene.tx, rx=scene.rx, f=f, grid=grid,
                                 precision=args.precision))
        if rf._HAS_CUPY:
            rf.cp.cuda.Stream.null.synchronize()
        return float(img.max())

    angs = np.arange(0.0, args.max_ang + 1e-9, args.step)
    r0 = peak(q)
    yaw_db, pit_db = [], []
    for a in angs:
        yaw_db.append(20 * np.log10(peak(rot(q, a, "z", ctr)) / r0))
        pit_db.append(20 * np.log10(peak(rot(q, a, "x", ctr)) / r0))
        print(f"  {a:4.2f} deg: yaw {yaw_db[-1]:6.2f} dB   pitch {pit_db[-1]:6.2f} dB")
    yaw_db, pit_db = np.array(yaw_db), np.array(pit_db)

    def width(y):
        b = np.where(y <= -3.0)[0]
        return float(np.interp(-3.0, y[b[0]-1:b[0]+1][::-1], angs[b[0]-1:b[0]+1][::-1])) \
            if len(b) and b[0] else np.nan
    wy, wp = width(yaw_db), width(pit_db)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    ax.plot(angs, yaw_db, "o-", label=f"YAW (azimuth)  -3dB @ {wy:.2f}°")
    ax.plot(angs, pit_db, "s-", label=f"PITCH (elevation)  -3dB @ {wp:.2f}°")
    ax.axhline(-3, ls=":", color="0.6")
    ax.set_xlabel("plate tilt (deg)"); ax.set_ylabel("specular peak return (dB rel. 0°)")
    ax.set_title("Specular falloff: azimuth vs elevation (flat plate)\n"
                 "similar curves => 2-D aperture handles elevation correctly")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "elevation_probe.png"); plt.close(fig)

    lines = [f"Elevation vs azimuth specular falloff -- plate {args.size*100:.0f} cm, {len(q)} pts",
             f"YAW (azimuth)  -3 dB half-width: {wy:.2f} deg",
             f"PITCH (elevation) -3 dB half-width: {wp:.2f} deg",
             f"ratio pitch/yaw: {wp/wy:.2f}  (near 1 => symmetric/handled; >>1 => elevation soft/degenerate)"]
    (out / "elevation_probe.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nDone. -> {out}")


if __name__ == "__main__":
    main()
