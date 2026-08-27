"""The standard simulation frame, stated once so no driver has to re-derive it.

    +z  into the scene, z = 0 AT THE ARRAY   (range)
    +y  up
    +x  to the LEFT as seen from the array
    right-handed:  x cross y = z

This is not a new convention. It is written verbatim in the MEASURED-data pipeline --
`reflect_array_imaging/processing/bistatic_recon.py` -- so adopting it here means a simulated scan
and a DK3 scan can be overlaid without a mental coordinate transform. It is also the frame in which
the retro floor panel becomes an unambiguous y = const plane, which is the bug that pipeline has
been carrying since 2026-07-31.

PLOTTING RULE: because +x is the array's LEFT and the viewer stands at the array, ALWAYS draw +x
to the left (`ax.invert_xaxis()`). Figures made this way are mirrored relative to anything produced
before 2026-08-26.

WHAT CHANGED FROM THE LEGACY SIM FRAME.  The old drivers used (x = array-right, y = RANGE,
z = vertical). Converting is a column swap plus one sign flip -- determinant +1, so right-handed is
preserved and this is a relabelling, not a mirror:

    legacy (x, y=range, z=vert)  ->  world (-x, z, y)

PROVEN SAFE.  The forward model and backprojection are frame-agnostic: `backproject` consumes
`grid.voxels()` as an (N,3) point set and the kernel only ever forms Euclidean distances. Measured
2026-08-26 (`phase0*.py`): permuting the frame changes the result by 5.99e-08 in fp64, i.e. nothing
in the chain is axis-aware. In fp32 it moves by ~1e-3 relative (0.0088 dB) because this is
interferometry at lambda~1 mm over r~2 m -- about 2000 wavelengths -- so fp32's ~1e-7 relative
precision becomes 2*pi*2000*1e-7 rad of phase. That is round-off, not a bug:

  * DO NOT validate a migration by bit-comparison in fp32. Use fp64, or a 1e-3 tolerance.
  * The world frame is in fact marginally MORE accurate in fp32 (2.213e-3 vs 2.284e-3 against fp64
    truth) because putting range in the LAST slot of dx^2+dy^2+dz^2 adds the two small terms before
    the large one swamps them.

SMPL-X.  The body model is x = the BODY's left, y = up, z = forward (out of the chest, i.e. TOWARD
the array when the subject faces it). `anatomy.py` states this at its `_FACING_AXIS` table and the
joints agree (left_hip x=+0.057, right_hip x=-0.057). Site labels are therefore ANATOMICAL and
view-independent -- a site is (region-from-LBS-weights, facing, height band) on fixed topology, so
`thigh_outer_l` is the person's left thigh in every view. Verified on both models, all 15 l/r pairs.
Note the consequence: the person's left is +x in the BODY frame but appears at NEGATIVE world x in a
FRONT view (they face you, sides swap) and POSITIVE in a BACK view. `assert_site_side` checks it.
"""
import numpy as np

X_AXIS, Y_AXIS, Z_AXIS = 0, 1, 2
AXIS_ROLE = {0: "x (left)", 1: "y (up)", 2: "z (range, into scene)"}


def smplx_to_world(P, yaw_deg, y_center, standoff, is_dir=False):
    """SMPL-X body coords -> world frame.

    P        : (N,3) positions, or directions when is_dir
    yaw_deg  : viewing yaw about the body's vertical axis (0 front, 180 back, +-90 sides)
    y_center : the body's vertical midpoint, so the subject is centred on y = 0
    standoff : array-to-subject distance; the subject lands near z = standoff
    """
    P = np.asarray(P, float)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    th = np.radians(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    x, z = c * x - s * z, s * x + c * z        # rotate about the body's vertical (y)
    if is_dir:
        return np.column_stack([-x, y, -z])
    return np.column_stack([-x, y - y_center, standoff - z])


def legacy_to_world(P, is_dir=False):
    """Legacy sim frame (x=array-right, y=range, z=vertical) -> world. det = +1."""
    P = np.asarray(P, float)
    return np.column_stack([-P[:, 0], P[:, 2], P[:, 1]])


def world_to_legacy(P, is_dir=False):
    """Inverse of `legacy_to_world` -- for reading pre-2026-08-26 data into the new frame."""
    P = np.asarray(P, float)
    return np.column_stack([-P[:, 0], P[:, 2], P[:, 1]])


def infer_frame(xs, ys, zs, standoff=2.0, tol=0.35):
    """Which axis holds range? Decided by which one sits near the standoff, not by name."""
    mids = [float(np.mean(a)) for a in (xs, ys, zs)]
    near = [i for i, m in enumerate(mids) if abs(m - standoff) < tol]
    if len(near) != 1:
        return "unknown", mids
    return ("world" if near[0] == Z_AXIS else "legacy" if near[0] == Y_AXIS else "unknown"), mids


def assert_frame(z, standoff=2.0, expect="world"):
    """Raise unless a loaded npz (or an (xs,ys,zs) tuple) is in the expected frame.

    Old files remain perfectly readable -- they carry their own axes -- but code written for the
    world frame will silently mis-index them, so every reader should call this.
    """
    xs, ys, zs = (z["xs"], z["ys"], z["zs"]) if hasattr(z, "keys") else z
    got, mids = infer_frame(xs, ys, zs, standoff)
    if got != expect:
        raise ValueError(
            f"frame mismatch: expected {expect!r}, found {got!r}. Axis means "
            f"x={mids[0]:+.3f} y={mids[1]:+.3f} z={mids[2]:+.3f} m; the range axis is the one near "
            f"standoff={standoff} m. Convert with frame.legacy_to_world() or read via load_scan().")
    return got


def assert_site_side(site, x_world, yaw_deg):
    """A `_l` site is the PERSON's left (+x in the body frame). Facing the array it lands at
    negative world x; seen from behind, positive. This is the sign error the migration can hide.

    x_world MUST be the SITE ANCHOR (its face centroid), not the object's centroid: an object is
    large next to the offset -- a 13 cm Glock seated at `wb_front_l` legitimately has its visible
    centroid on the far side of the midline -- so feeding it seat_xyz produces false alarms.
    """
    if not site.endswith(("_l", "_r")):
        return True
    want_body = +1 if site.endswith("_l") else -1
    expect = -want_body * np.sign(np.cos(np.radians(yaw_deg)))
    if expect == 0 or x_world == 0:
        return True                                    # side views: no left/right prediction
    if np.sign(x_world) != expect:
        raise ValueError(f"site {site!r} at yaw {yaw_deg}: world x={x_world:+.3f} but the person's "
                         f"{'left' if want_body>0 else 'right'} should give sign {expect:+.0f}")
    return True


def frame_summary(standoff=2.0):
    """Print this in every driver log and figure caption, as the measured pipeline does."""
    return ("Frame (right-handed): +z INTO the scene (z=0 at the array), +y up, +x LEFT.\n"
            f"  subject centred near z = {standoff:.2f} m, y = 0; plots draw +x to the LEFT.\n"
            "  matches reflect_array_imaging/processing/bistatic_recon.py.")


if __name__ == "__main__":
    print(frame_summary())
    M = np.array([[-1., 0, 0], [0, 0, 1], [0, 1, 0]])
    print(f"\nlegacy->world determinant = {np.linalg.det(M):+.1f}  (+1 = right-handed preserved)")
    p = np.array([[0.3, 1.9, 0.4]])                     # legacy: array-right, range, up
    print("legacy (x=+0.3 right, y=1.9 range, z=+0.4 up) -> world", legacy_to_world(p)[0])
    assert np.allclose(world_to_legacy(legacy_to_world(p)), p), "round-trip failed"
    print("round-trip OK")


# --------------------------------------------------------------------------------------------
# PHASE 5: reading data of either vintage.
# ds3-ds5b and the ds6 pilot were written in the legacy frame. They stay perfectly readable --
# every npz carries its own xs/ys/zs -- but world-frame code will silently mis-index them, so
# route every read through `load_scan`, which normalises to the world frame whatever the vintage.
# --------------------------------------------------------------------------------------------

_LEGACY_MAP = {  # stored key -> how to convert
    "han_db": "image", "depth_map": "image",
    "obj_vox": "vox", "gun_vox": "vox",
    "seat_xyz": "point", "obj_aabb": "aabb", "gun_aabb": "aabb",
}


def load_scan(path, standoff=2.0):
    """Load a scan and return every array in the WORLD frame, whatever frame it was written in.

    Legacy -> world is (x, y=range, z=vert) -> (-x, z, y), so on stored arrays it is a flip of the
    x axis plus a swap of the two non-x axes. `field_gated` is untouched: it is a per-cell range
    profile, and the range axis is range in both frames.
    """
    import numpy as np
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    got, _ = infer_frame(d["xs"], d["ys"], d["zs"], standoff)
    if got == "world":
        d["frame"] = "world_z_range_y_up_x_left"
        return d
    if got != "legacy":
        raise ValueError(f"{path}: cannot identify the frame from its axes")

    xs_l, ys_l, zs_l = np.asarray(d["xs"]), np.asarray(d["ys"]), np.asarray(d["zs"])
    nx, nv = len(xs_l), len(zs_l)
    d["xs"] = (-xs_l)[::-1].astype(np.float32)     # x flips sign; reverse to stay ascending
    d["ys"] = zs_l.astype(np.float32)              # vertical
    d["zs"] = ys_l.astype(np.float32)              # range
    for k, kind in _LEGACY_MAP.items():
        if k not in d or d[k] is None:
            continue
        a = np.asarray(d[k])
        if kind == "image" and a.ndim == 2:
            d[k] = a[::-1, :]                                   # (x, vert) with x flipped
        elif kind == "vox" and a.size:
            ix, iy, iz = a[:, 0], a[:, 1], a[:, 2]              # iy was range, iz vertical
            d[k] = np.column_stack([(nx - 1) - ix, iz, iy]).astype(a.dtype)
        elif kind == "point" and a.size == 3 and np.isfinite(a).all():
            d[k] = legacy_to_world(np.asarray(a, float)[None, :])[0].astype(a.dtype)
        elif kind == "aabb" and a.shape == (2, 3) and np.isfinite(a).all():
            w = legacy_to_world(np.asarray(a, float))
            d[k] = np.vstack([w.min(0), w.max(0)]).astype(a.dtype)
    if "gate_cells" in d and np.asarray(d["gate_cells"]).size:
        gc = np.asarray(d["gate_cells"])
        ix, iv = np.unravel_index(gc, (nx, nv))                 # legacy flat over (x, vertical)
        d["gate_cells"] = (((nx - 1) - ix) * nv + iv).astype(gc.dtype)
    if "ext" in d:
        e = np.asarray(d["ext"], float)
        d["ext"] = np.array([-e[1], -e[0], e[2], e[3]], np.float32)
    d["frame"] = "world_z_range_y_up_x_left"
    d["frame_converted_from"] = "legacy"
    return d
