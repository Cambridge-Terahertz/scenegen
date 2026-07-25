"""Seat a threat on a posed SMPL-X body -- invariant to pose AND shape.

A placement is (face, bary, roll, standoff) in SMPL-X TOPOLOGY space. To realize it
on any posed/shaped body we need a local frame at that face that DEFORMS WITH THE
BODY. The trick:

  * the face's own normal n and its edge-derived tangent come free from the posed
    verts (LBS carries them), but edge-0 is an arbitrary triangulation artifact, so
    a roll measured from it is meaningless;
  * so we measure, ON THE CANONICAL MESH, the angle `phi` between the anatomical
    up-direction (SMPL-X +y, projected into the face's tangent plane) and edge-0;
  * on the posed mesh we rebuild the edge-0 frame and rotate by that same `phi`.

The result is an anatomical "along-the-body" axis transported through the face's own
deformation. roll=0 therefore means "long axis along the limb/torso" for every body
in every pose -- a gun on the thigh swings with the femur, for free.

All work happens in the SMPL-X frame (x=left, y=up, z=forward). Map body+threat to
the sim/portal world frame TOGETHER, afterwards.
"""
from __future__ import annotations
import numpy as np


def _unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-30)


def face_frame(V, F, face, bary):
    """(p, n, e_t, e_u) for one face of a vert set: point, outward normal, and an
    orthonormal tangent pair built from edge 0."""
    tri = V[F[face]]                                   # (3,3)
    p = np.asarray(bary, np.float64) @ tri
    e0, e1 = tri[1] - tri[0], tri[2] - tri[0]
    n = _unit(np.cross(e0, e1))
    t = _unit(e0 - (e0 @ n) * n)
    u = np.cross(n, t)
    return p, n, t, u


def anatomical_roll_ref(Vc, F, face, up=(0.0, 1.0, 0.0)):
    """Angle from the face's edge-0 tangent to the anatomical up-direction, measured
    ON THE CANONICAL MESH. This is the constant that makes roll meaningful."""
    _, n, t, u = face_frame(Vc, F, face, (1 / 3, 1 / 3, 1 / 3))
    a = np.asarray(up, np.float64)
    a = a - (a @ n) * n
    if np.linalg.norm(a) < 1e-6:                       # face is ~horizontal (shoulder
        a = t                                          # top, foot): fall back to edge-0
    a = _unit(a)
    return float(np.arctan2(a @ u, a @ t))


def seat_frame(Vp, Vc, F, face, bary, roll_deg=0.0):
    """The body-carried frame to seat an object into, on the POSED verts Vp.
    Returns (p, R) where R's columns map object axes (long, mid, thin) -> world.
    Object +z (thin) ends up along the outward surface normal."""
    p, n, t, u = face_frame(Vp, F, face, bary)
    phi = anatomical_roll_ref(Vc, F, face) + np.deg2rad(roll_deg)
    a = np.cos(phi) * t + np.sin(phi) * u              # anatomical long axis, rolled
    b = np.cross(n, a)
    return p, np.column_stack([a, b, n])               # (3,3), det +1


def attach_rigid(obj_pts, obj_nrm, Vp, Vc, F, face, bary, *, roll_deg=0.0,
                 standoff=0.012, half_thick=None):
    """Rigidly seat an object (in canonical object frame) flat against the surface.
    standoff = clothing gap between skin and the object's inner face [m]."""
    p, R = seat_frame(Vp, Vc, F, face, bary, roll_deg)
    if half_thick is None:
        half_thick = 0.5 * np.ptp(obj_pts[:, 2])
    q = obj_pts @ R.T + p + R[:, 2] * (half_thick + standoff)
    n = obj_nrm @ R.T if obj_nrm is not None else np.tile(R[:, 2], (len(q), 1))
    return q, n


def attach_conformal(obj_pts, obj_nrm, Vp, Vc, F, face, bary, *, roll_deg=0.0,
                     standoff=0.012, vert_normals=None, k_search=4000):
    """Bend a slab/wire onto the body surface (a sheet of explosive taped to a
    curved torso does NOT sit flat). Each object point is pushed out along the local
    body normal from its nearest body-surface vertex, preserving its through-thickness
    coordinate. Needs per-vertex normals of the POSED mesh."""
    from scipy.spatial import cKDTree
    p, R = seat_frame(Vp, Vc, F, face, bary, roll_deg)
    thin_lo = obj_pts[:, 2].min()
    depth = obj_pts[:, 2] - thin_lo                    # 0 at the inner face
    # lay the object out in the tangent plane at the seat, then re-project
    flat = obj_pts.copy()
    flat[:, 2] = 0.0
    world_flat = flat @ R.T + p

    # restrict the search to body verts near the seat (cheap + avoids wrapping to
    # the far side of a limb)
    d2 = np.sum((Vp - p) ** 2, axis=1)
    near = np.argpartition(d2, min(k_search, len(Vp) - 1))[:k_search]
    tree = cKDTree(Vp[near])
    _, j = tree.query(world_flat, k=1, workers=-1)
    base = Vp[near][j]
    nb = _unit(vert_normals[near][j])
    q = base + nb * (standoff + depth)[:, None]
    return q, nb


def front_cull(q, n, s, view_dir=(0.0, -1.0, 0.0)):
    """Drop object points whose outward normal faces AWAY from the array.

    Needed because `bodies.zbuffer_cull` keeps everything within `keep_depth`=2 cm of
    the front surface -- a depth tuned for a ~30 cm-deep BODY. A pistol is 3 cm thick
    and a knife 4 mm, so a thin threat lies ENTIRELY inside that window and its
    occluded BACK face survives the cull and radiates too. That double-counts the
    threat (measured: 73% of a closed Glock survived, vs the ~50% a closed surface
    should give). Culling on the seated world normals is the physically right fix, and
    is only possible now that the assets carry real normals.

    view_dir : direction from the target TOWARD the array. In the sim frame the array
               sits at y=0 and the body at y=+standoff, so that is -y.
    """
    vis = (np.asarray(n, np.float64) @ np.asarray(view_dir, np.float64)) > 0.0
    return q[vis], n[vis], s[vis]


#: 180-deg rotation about the object's LONG axis (+x). Not a mirror -- det = +1.
_FLIP = np.array([1.0, -1.0, -1.0])


def attach(threat, Vp, Vc, F, placement, *, target_nn, vert_normals=None,
           rng=None, standoff=0.012, roll_jitter=0.0):
    """Full path: sample the threat, seat it per `placement`. -> (q, n, s).

    `s` comes from the threat's own per-part material map (the Glock is a steel slide
    on a polymer frame), so it is per-point, not a single scalar.

    placement["flip"] turns the object over about its own long axis, which swaps WHICH FACE
    lies against the body and reverses the grip direction while LEAVING THE MUZZLE WHERE IT IS.
    That is a distinct degree of freedom from `roll`, which spins about the surface normal and
    would send a muzzle-down carry muzzle-UP. It is what makes a carry handed: the seat frame
    b = n x a flips sign between a left (+90 deg) and a right (-90 deg) seat, so an unmirrored
    asset comes out grip-forward on one leg and grip-back on the other. Flip the one that is
    wrong rather than mirroring the asset -- mirroring a chiral scan would make it a left-handed
    Glock, which is not a thing."""
    obj_q, obj_n, obj_s = threat.sample(target_nn)
    if placement.get("flip", False):
        obj_q = obj_q * _FLIP
        if obj_n is not None:
            obj_n = obj_n * _FLIP
    roll = placement.get("roll", 0.0)
    if roll_jitter and rng is not None:
        roll += rng.normal(0, roll_jitter)
    kw = dict(roll_deg=roll, standoff=placement.get("standoff", standoff))
    if threat.conformal:
        if vert_normals is None:
            raise ValueError("conformal attach needs posed vert_normals")
        q, n = attach_conformal(obj_q, obj_n, Vp, Vc, F, placement["face"],
                                placement["bary"], vert_normals=vert_normals, **kw)
    else:
        q, n = attach_rigid(obj_q, obj_n, Vp, Vc, F, placement["face"],
                            placement["bary"], **kw)
    return q, n, obj_s
