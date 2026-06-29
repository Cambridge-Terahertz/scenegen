"""Scatterer-cloud sources for the forward model.

Two entry points, with very different status:

* ``load_xyz``      -- REAL data path. Port of read_xyz_binary.m: loads an
                       actual exported body/asset point cloud. Production.
* ``synthetic_body`` -- TEST FIXTURE ONLY. A crude parametric body of ellipsoid
                       primitives, used to exercise forward->recon->save at real
                       array scale and to get timing. It is NOT the model
                       generation pipeline -- the real pose/asset flow (Mixamo
                       export -> place_assets -> HPR cull, simulator_mixamo_flow.m)
                       is a separate subsystem that has not been ported. Do not
                       treat synthetic_body output as physically realistic.

Geometry convention matches the rest of the pipeline: x = width, y = depth
(downrange; the array sits near y=0 and the body stands at y=standoff),
z = height. Only front-facing surface points (facing the array, -y) are kept,
loosely mimicking the HPR occlusion cull in the MATLAB flow.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["synthetic_body", "load_xyz", "load_real_scene",
           "density_weights", "zbuffer_cull"]


def zbuffer_cull(q, pixel: float = 0.001, depth_axis: int = 1) -> np.ndarray:
    """Keep only the nearest point per cross-range pixel along `depth_axis`,
    i.e. the front surface facing the array.

    The array sits at the near side and cannot see through the body, so the
    occluded back surface (~half the points) must not contribute. This is a
    simple z-buffer (port of the role of ZBuffer2.m): bin in the two cross-range
    axes, keep the smallest-depth point per bin. Returns kept indices.
    """
    q = np.asarray(q)
    other = [a for a in range(3) if a != depth_axis]
    cell = np.floor(q[:, other] / pixel).astype(np.int64)
    cell -= cell.min(0)
    key = cell[:, 0] * (cell[:, 1].max() + 1) + cell[:, 1]
    order = np.argsort(q[:, depth_axis], kind="stable")   # nearest (small depth) first
    first = np.unique(key[order], return_index=True)[1]
    return order[first]


def density_weights(q, voxel: float = 0.005) -> np.ndarray:
    """Per-point weight ~ 1/(local point density), so each occupied voxel of
    surface contributes equally regardless of how finely it was meshed.

    Mesh exporters sample geometric detail (face, hands, folds) more densely;
    without this, the forward sum over-weights those regions in the RF image.
    Port of compute_density_weights_gpu.m. Returns weights scaled to mean 1.
    """
    g = np.floor(np.asarray(q, np.float64) / voxel).astype(np.int64)
    g -= g.min(0)
    dims = g.max(0) + 1
    key = (g[:, 0] * dims[1] + g[:, 1]) * dims[2] + g[:, 2]
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    return w * (len(w) / w.sum())   # scale to mean 1 (relative weighting is what matters)


def load_xyz(path) -> np.ndarray:
    """Read a raw float32 XYZ point cloud (port of read_xyz_binary.m).

    The file is interleaved float32 triplets; returns (N, 3)."""
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(-1, 3)


def load_real_scene(folder, *, decimate: int = 1, assets=("phone",),
                    standoff: float = 2.0, zbuffer: bool = True,
                    zbuffer_pixel: float = 0.0002, density_weight: bool = True,
                    density_voxel: float = 0.005):
    """Load `<folder>/body/body.xyz` + placed asset clouds and put them in the
    simulation frame.

    The exported clouds are in an upright "export frame" (x=width, y=height,
    z=depth). This reorients to the sim frame (x=width, **y=depth** at `standoff`
    with the front facing the array, z=height centred), decimates the body by
    `decimate`, and tags reflectivity from the body/asset JSONs.

    Only assets already placed in the body frame should be passed (e.g. `phone`);
    unplaced assets (e.g. `gun`, exported at the origin) need the place_assets
    positioning, which is not ported. Returns (q, s).
    """
    folder = Path(folder)
    body = load_xyz(folder / "body" / "body.xyz")
    body_s = json.loads((folder / "body" / "body.json").read_text())["body"]["s"]

    # centering offsets derived from the body, applied identically to assets so
    # their placement relative to the body is preserved
    y_mean = float(body[:, 1].mean())   # height -> centred at 0
    z_mean = float(body[:, 2].mean())   # depth  -> centred at standoff

    def reorient(p):
        return np.column_stack([
            p[:, 0],                          # width  (x)
            standoff - (p[:, 2] - z_mean),    # depth  (front, +z_export, toward array)
            p[:, 1] - y_mean,                 # height (z), centred
        ])

    qb = reorient(body)[::decimate]
    q_list, s_list = [qb], [np.full(len(qb), body_s, np.float64)]

    if assets:
        ajson = folder / "assets" / "assets.json"
        asset_s = json.loads(ajson.read_text()) if ajson.exists() else {}
        for name in assets:
            f = folder / "assets" / f"{name}.xyz"
            if not f.exists() or name not in asset_s:
                print(f"  (asset '{name}' not available, skipping)")
                continue
            pa = reorient(load_xyz(f))
            q_list.append(pa)
            s_list.append(np.full(len(pa), asset_s[name]["s"], np.float64))
            print(f"  + asset '{name}': {len(pa):,} pts, s={asset_s[name]['s']}")

    q = np.vstack(q_list)
    s = np.concatenate(s_list)          # material reflectivity per point
    if zbuffer:
        keep = zbuffer_cull(q, zbuffer_pixel)
        q, s = q[keep], s[keep]          # front surface only (occlusion)
        print(f"  z-buffer cull (pixel {zbuffer_pixel*1e3:.1f} mm): "
              f"{len(keep):,} front-surface pts")
    if density_weight:
        w = density_weights(q, density_voxel)
        s = s * w                        # area-normalise non-uniform meshing
        print(f"  density-weighted (voxel {density_voxel*1e3:.0f} mm)")
    return q, s


def _ellipsoid_front(center, radii, n, rng, oversample=3.0):
    """Sample ~n points on the front (-y facing) surface of an ellipsoid."""
    m = int(n * oversample)
    v = rng.normal(size=(m, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)      # unit sphere
    p = np.asarray(center) + v * np.asarray(radii)     # ellipsoid surface
    normal = v / np.asarray(radii)                     # outward normal direction
    front = normal[:, 1] < 0                           # facing the array (-y)
    p = p[front]
    return p[:n] if len(p) >= n else p


def synthetic_body(n_points: int = 200_000, *, standoff: float = 2.0,
                   with_asset: bool = True, seed: int = 0):
    """A body-shaped front-surface cloud + optional concealed asset.

    Returns (q, s): positions (N,3) and per-point reflectivity (N,).
    """
    rng = np.random.default_rng(seed)
    y = standoff
    # (name, center, radii, fraction of points)
    parts = [
        ("torso", (0.00, y, 0.20), (0.18, 0.13, 0.34), 0.34),
        ("head",  (0.00, y, 0.66), (0.10, 0.10, 0.12), 0.08),
        ("arm_l", (-0.26, y, 0.08), (0.06, 0.06, 0.30), 0.12),
        ("arm_r", (0.26, y, 0.08), (0.06, 0.06, 0.30), 0.12),
        ("leg_l", (-0.10, y, -0.55), (0.09, 0.09, 0.42), 0.17),
        ("leg_r", (0.10, y, -0.55), (0.09, 0.09, 0.42), 0.17),
    ]
    pts, refl = [], []
    for _name, c, r, frac in parts:
        p = _ellipsoid_front(c, r, max(int(n_points * frac), 1), rng)
        pts.append(p)
        refl.append(np.full(len(p), 0.7))              # body reflectivity

    if with_asset:
        # a small dense bright patch on the torso front (gun-like)
        na = max(int(n_points * 0.02), 200)
        cx, cz = 0.08, -0.05
        ax = np.column_stack([
            cx + rng.uniform(-0.06, 0.06, na),
            (y - 0.13) + rng.uniform(-0.004, 0.004, na),   # just in front of torso
            cz + rng.uniform(-0.09, 0.09, na),
        ])
        pts.append(ax)
        refl.append(np.full(na, 1.0))                  # high reflectivity

    q = np.vstack(pts)
    s = np.concatenate(refl)
    return q, s
