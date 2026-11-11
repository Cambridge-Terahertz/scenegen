"""Occlusion (hidden-point-removal) culling, with a selectable method.

Two implementations:
  - "convex_hull": the original Katz-Tal-Basri spherical-flip + ConvexHull
    algorithm (hpr.hpr()). Gets slow and memory-heavy on very large point
    clouds, since the spherical-flip transform pushes nearly all points onto
    a huge, near-degenerate hull that qhull has to fully resolve.
  - "zbuffer": sparse angular depth-buffer. Bins points by (azimuth,
    elevation) relative to the camera; within each bin, keeps every point
    within `zbuffer_depth_tolerance_m` of that bin's nearest point (not just
    the single nearest one). Memory scales with the number of *occupied*
    bins, not hull complexity -- much cheaper for huge point clouds.

    An earlier version of this kept only the single nearest point per bin
    ("winner-take-all"). That doesn't work for densely, independently
    sampled point clouds (e.g. a real 3D scan): to also catch far-side
    occlusion you'd need bins coarse enough to lump together points from
    opposite sides of the object, but at that same coarseness the bins ALSO
    swallow many genuinely-different nearby points on the near surface --
    there's no single angular_res where both effects separate cleanly,
    since near-surface point spacing and "which points share a viewing
    direction" live at similar angular scales for continuous random
    sampling (unlike structured, one-sample-per-ray scan/depth-camera
    data). Verified empirically: coarsening angular_res on a synthetic
    sphere just monotonically drove the kept fraction toward 0 instead of
    settling near the ~50% a convex object should show.

    The fix is to separate the two concerns: angular_res only needs to be
    fine enough to distinguish real, different viewing directions (it can
    be coarse relative to point spacing -- matched to the imaging system's
    actual resolution, not the scan's), and a *depth* tolerance decides
    what counts as "the same surface" within a bin: points within
    `zbuffer_depth_tolerance_m` of the bin's closest point are kept (real
    near-surface detail/roughness), points farther than that are discarded
    (genuinely occluded, e.g. the opposite side of a body). Both parameters
    must be tuned against the actual point count/geometry -- see the
    diagnostic hpr_zbuffer() prints each call.

Selected via params["occlusion"]["method"] in the sim_params JSON. Defaults
to "convex_hull" (matching prior behavior) if the "occlusion" section, or
the "method" key within it, is absent.
"""
import numpy as np

from hpr import hpr as _hpr_convex_hull

DEFAULT_METHOD = "convex_hull"
DEFAULT_CONVEX_HULL_PARAM = 2.5
DEFAULT_ZBUFFER_ANGULAR_RES_RAD = 2e-3  # tune per point count -- see hpr_zbuffer's diagnostic
DEFAULT_ZBUFFER_DEPTH_TOLERANCE_M = 0.03  # tune to object surface relief -- see hpr_zbuffer's diagnostic
DEFAULT_FORWARD = (0.0, 1.0, 0.0)


def hpr_zbuffer(p, camera, forward=DEFAULT_FORWARD,
                 angular_res=DEFAULT_ZBUFFER_ANGULAR_RES_RAD,
                 depth_tolerance=DEFAULT_ZBUFFER_DEPTH_TOLERANCE_M):
    """Sparse angular depth-buffer occlusion culling.

    Bins points by (azimuth, elevation) as seen from `camera` looking along
    `forward`; within each bin, keeps all points within `depth_tolerance` of
    that bin's nearest point (preserving near-surface detail) and discards
    the rest (genuinely occluded, e.g. the far side of the object).

    p: (N, 3) point cloud
    camera: (3,) viewpoint position
    forward: (3,) direction the camera looks along (need not be normalized)
    angular_res: bin width in radians, applied to both azimuth and
        elevation -- should match the imaging system's actual angular
        resolution, not the point cloud's raw spacing (which would make
        bins far finer than needed and mostly singleton)
    depth_tolerance: max distance (meters) behind a bin's nearest point for
        another point in that bin to still count as visible surface,
        rather than occluded

    Returns indices into p of the visible (non-occluded) points.
    """
    p = np.asarray(p, dtype=np.float64)
    camera = np.asarray(camera, dtype=np.float64)

    fwd = np.asarray(forward, dtype=np.float64)
    fwd = fwd / np.linalg.norm(fwd)

    # Build an arbitrary right/up basis orthogonal to fwd.
    tmp = np.array([0.0, 0.0, 1.0]) if abs(fwd[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(fwd, tmp)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    rel = p - camera
    r = np.linalg.norm(rel, axis=1)

    x_loc = rel @ right
    y_loc = rel @ up
    z_loc = rel @ fwd  # forward (depth) component

    az = np.arctan2(x_loc, z_loc)
    el = np.arctan2(y_loc, np.hypot(x_loc, z_loc))

    az_bin = np.floor(az / angular_res).astype(np.int64)
    el_bin = np.floor(el / angular_res).astype(np.int64)

    # Pack (az_bin, el_bin) into one collision-free key, sized exactly to
    # the data's actual bin range (not a fixed/dense grid).
    az_shift = az_bin - az_bin.min()
    el_shift = el_bin - el_bin.min()
    el_range = int(el_shift.max()) + 1
    keys = az_shift * el_range + el_shift

    # Sort by (key, range) so each key-run is ordered near-to-far; the first
    # row of a run is the bin's nearest point.
    order = np.lexsort((r, keys))
    sorted_keys = keys[order]
    sorted_r = r[order]

    n = len(sorted_keys)
    is_group_start = np.empty(n, dtype=bool)
    if n > 0:
        is_group_start[0] = True
        is_group_start[1:] = sorted_keys[1:] != sorted_keys[:-1]

    n_bins = int(is_group_start.sum())

    # Broadcast each group's minimum range to every member of that group.
    group_id = np.cumsum(is_group_start) - 1
    r_min_per_group = sorted_r[is_group_start]
    r_min_broadcast = r_min_per_group[group_id]

    keep_sorted = sorted_r <= (r_min_broadcast + depth_tolerance)

    n_points = len(p)
    n_kept = int(keep_sorted.sum())
    avg_per_bin = n_points / n_bins if n_bins else 0.0
    print(f"hpr_zbuffer: {n_points} points, {n_bins} occupied angular bins "
          f"({avg_per_bin:.2f} points/bin on average), depth_tolerance="
          f"{depth_tolerance:g} m -> kept {n_kept} ({n_kept/n_points:.1%}). "
          f"If avg points/bin is close to 1.0, angular_res={angular_res:g} "
          f"is too fine for this point density (little for depth_tolerance "
          f"to work with); if the kept fraction still looks too "
          f"high/low, adjust depth_tolerance (larger discards more of the "
          f"far side, smaller keeps more of it).")

    visible_inds = order[keep_sorted]
    return np.sort(visible_inds)


def cull_occluded(p, camera, occlusion_params=None):
    """Dispatch to the occlusion-culling method configured in the sim_params
    JSON's "occlusion" section (params["occlusion"]).

    occlusion_params: that dict (or None/{} to fall back to all defaults).
    """
    occlusion_params = occlusion_params or {}
    method = occlusion_params.get("method", DEFAULT_METHOD)

    if method == "convex_hull":
        param = occlusion_params.get("convex_hull_param", DEFAULT_CONVEX_HULL_PARAM)
        return _hpr_convex_hull(p, camera, param)
    elif method == "zbuffer":
        forward = occlusion_params.get("forward", list(DEFAULT_FORWARD))
        angular_res = occlusion_params.get("zbuffer_angular_res_rad", DEFAULT_ZBUFFER_ANGULAR_RES_RAD)
        depth_tolerance = occlusion_params.get("zbuffer_depth_tolerance_m", DEFAULT_ZBUFFER_DEPTH_TOLERANCE_M)
        return hpr_zbuffer(p, camera, forward=forward, angular_res=angular_res, depth_tolerance=depth_tolerance)
    else:
        raise ValueError(
            f"Unknown params['occlusion']['method']: {method!r} "
            "(expected 'convex_hull' or 'zbuffer')"
        )
