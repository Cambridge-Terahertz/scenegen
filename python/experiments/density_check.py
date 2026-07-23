"""Validate the density correction (bodies.density_weights).

density_weights bins points into `voxel` (5 mm) 3-D cells and weights each point
1/(points in its cell), so every OCCUPIED VOXEL contributes equally regardless of
how finely it was meshed. That is a *proxy* for true surface area. This script
stresses the proxy on known geometry where the true area is analytic:

  T1  recovery      -- two equal-area halves sampled at different density; does the
                       correction turn the biased point-count back into 50/50?
  T2  orientation   -- equal-area plates tilted vs the voxel grid; a plane crosses
                       more voxels as it tilts, so per-voxel weighting OVER-weights
                       oblique surfaces (bias up to ~sqrt(3)). Quantify it.
  T3  fold/thickness-- two surface sheets closer than one voxel merge into one cell
                       and get counted as HALF the area. Quantify vs separation.

Pure numpy, no GPU. Real-body characterisation is in density_check_body (needs the
.xyz + GPU).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bodies import density_weights   # noqa: E402


def frac_weight(mask, w):
    return 100.0 * w[mask].sum() / w.sum()


def t1_recovery(voxel=0.005):
    """Flat plate [0,1]x[0,1], z=0. Left half 4x denser than right. True area 50/50."""
    rng = np.random.default_rng(0)
    step_dense, step_sparse = 0.0025, 0.005          # point spacing (m)
    xl, yl = np.meshgrid(np.arange(0, 0.5, step_dense), np.arange(0, 1, step_dense))
    xr, yr = np.meshgrid(np.arange(0.5, 1, step_sparse), np.arange(0, 1, step_sparse))
    ql = np.column_stack([xl.ravel(), yl.ravel(), np.zeros(xl.size)])
    qr = np.column_stack([xr.ravel(), yr.ravel(), np.zeros(xr.size)])
    q = np.vstack([ql, qr]); left = np.arange(len(q)) < len(ql)
    w = density_weights(q, voxel)
    print("T1 recovery (true 50/50):")
    print(f"   points     left {100*left.mean():5.1f}%   right {100*(~left).mean():5.1f}%  (biased)")
    print(f"   corrected  left {frac_weight(left, w):5.1f}%   right {frac_weight(~left, w):5.1f}%  (target 50)")


def _occupied_voxels(q, voxel):
    """Effective-area proxy: number of distinct voxels the point set touches."""
    g = np.floor(q / voxel).astype(np.int64)
    return len(np.unique(g[:, 0] * 1_000_003 * 1_000_003 + g[:, 1] * 1_000_003 + g[:, 2]))


def _plate(normal, area_side=0.4, npts=400_000, seed=0):
    """Square plate of fixed true area, oriented so its surface normal == `normal`,
    sampled area-uniformly at RANDOM. Random (not gridded) sampling is essential: a
    regular point lattice rotated by a rational angle resonates with the voxel grid and
    the occupied-voxel count reflects that commensurability, not the geometry."""
    n = np.asarray(normal, float); n = n / np.linalg.norm(n)
    tmp = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    uv = np.random.default_rng(seed).uniform(0, area_side, size=(npts, 2))
    return uv[:, 0:1] * e1 + uv[:, 1:2] * e2


def t2_orientation(voxel=0.005):
    """Two equal-true-area plates in ONE cloud -- a flat reference (normal z-hat) and a
    tilted one (normal swept from z-hat toward the body diagonal) -- weighted TOGETHER by
    a single density_weights() call. A region's recon brightness is proportional to its
    share of the total weight, so for equal area the tilted plate should match the flat
    one (ratio 1.00). Plain occupancy over-weights the tilted plate by |n|_1 (up to
    sqrt(3)); the normal fallback cancels it. Off-axis azimuth (phi=35) avoids the
    single-coordinate-axis 45 deg degeneracy (y==z) that collapses the count."""
    print("T2 orientation over-weighting (tilted/flat weight ratio; ideal = 1.00x):")
    print(f"   {'normal polar':>12}   {'|n|_1':>6}   {'raw occupancy':>13}   {'n-fallback':>10}")
    phi = np.radians(35.0)
    flat = _plate([0, 0, 1]) + np.array([2.0, 0, 0])         # reference, moved clear
    nflat = np.tile([0.0, 0.0, 1.0], (len(flat), 1))
    for a_deg in (0, 15, 30, 45, 54.7, 60):
        a = np.radians(a_deg)
        n = np.array([np.sin(a) * np.cos(phi), np.sin(a) * np.sin(phi), np.cos(a)])
        tilt = _plate(n)
        q = np.vstack([flat, tilt])
        nn = np.vstack([nflat, np.tile(n, (len(tilt), 1))])
        is_t = np.arange(len(q)) >= len(flat)
        raw = density_weights(q, voxel)                      # old: occupancy only
        fix = density_weights(q, voxel, normals=nn)          # new: |n|_1 fallback
        r_raw = raw[is_t].sum() / raw[~is_t].sum()
        r_fix = fix[is_t].sum() / fix[~is_t].sum()
        print(f"   {a_deg:9.1f} deg   {np.abs(n).sum():6.2f}   {r_raw:12.2f}x   {r_fix:9.2f}x")


def t3_fold(voxel=0.005):
    """Two parallel sheets separated by `sep`; true area = 2x one sheet. When sep<voxel
    they share cells -> counted as ~1 sheet."""
    print("T3 fold/thickness under-count (true area = 2.00x a single sheet):")
    u = np.arange(0, 0.3, 0.002); uu, vv = np.meshgrid(u, u)
    one = np.column_stack([uu.ravel(), vv.ravel(), np.zeros(uu.size)])
    n1 = _occupied_voxels(one, voxel)
    for sep in (0.001, 0.003, 0.005, 0.010, 0.020):
        two = np.vstack([one, one + np.array([0, 0, sep])])
        print(f"   sep {sep*1e3:4.0f} mm : occupied voxels {_occupied_voxels(two, voxel)/n1:4.2f}x one sheet  "
              f"(true 2.00x){'   <-- merged' if sep < voxel else ''}")


if __name__ == "__main__":
    t1_recovery(); print()
    t2_orientation(); print()
    t3_fold()
