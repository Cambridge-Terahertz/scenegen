"""SMPL-X anatomy: part labels + carry sites, defined in TOPOLOGY space.

The whole point: SMPL-X topology is fixed (10475 verts / 20908 faces) for every
shape (betas) and every pose. So an anatomical location is a FACE INDEX, not a
world coordinate. A carry site is therefore a set of candidate faces on the
canonical mesh; sampling a site gives (face, barycentric) which lands on the SAME
anatomical spot for any body, in any pose. See attach.py for the frame that rides
along with it.

Part labels come from the model's own linear-blend-skinning weights (argmax over
the 55 joints) -- no external segmentation file needed.
"""
from __future__ import annotations
import numpy as np

# SMPL-X body joints 0..24 (the rest are hands/face)
JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "jaw", "left_eye", "right_eye",
]
NAME2J = {n: i for i, n in enumerate(JOINT_NAMES)}

# Which joints make up each anatomical region we might conceal something on.
REGIONS = {
    "waistband":   ["pelvis"],                       # belt line / appendix carry
    "abdomen":     ["spine1", "spine2"],             # midriff
    "chest":       ["spine3", "left_collar", "right_collar"],
    "torso":       ["pelvis", "spine1", "spine2", "spine3"],
    "left_thigh":  ["left_hip"],
    "right_thigh": ["right_hip"],
    "left_shin":   ["left_knee"],
    "right_shin":  ["right_knee"],
    "left_ankle":  ["left_ankle"],
    "right_ankle": ["right_ankle"],
    "left_upper_arm":  ["left_shoulder"],
    "right_upper_arm": ["right_shoulder"],
    "left_forearm":    ["left_elbow"],
    "right_forearm":   ["right_elbow"],
    "lower_back":  ["pelvis", "spine1"],             # + back-facing filter
    "upper_back":  ["spine2", "spine3"],             # + back-facing filter
    "beltline":    ["pelvis", "spine1"],             # + azimuth filter + BELT height band
}

# The belt ring, in canonical-mesh height (SMPL-X +y = up). Measured, not guessed: this is the
# only band where ALL EIGHT 45-deg azimuth sectors are populated (>=20 faces each, symmetric
# l/r). Below it the hips flare and the sides disappear; above it the pelvis+spine1 region ends.
# NB the legacy `appendix` site's area-centroid sits ~6 cm LOWER (y=-36 cm) because the pelvis
# region runs all the way down to the crotch and drags the centroid with it -- so `wb_front_mid`
# is NOT the same seat as `appendix`. Both are kept; `appendix` remains the walkthrough anchor.
BELT = (-0.35, -0.25)

# The ankle-holster band, likewise measured -- and note it lives on the SHIN region, not the
# ankle one. Two things forced that:
#   1. the `left_ankle`/`right_ankle` LBS regions INCLUDE THE SOLE OF THE FOOT. With
#      facing="any" the area-centroid face came out at normal (0.11, -0.99, 0.07) -- straight
#      DOWN -- so the pistol was seated under the foot. A horizontal face also makes the roll
#      reference degenerate (anatomical "up" projects to nothing in its tangent plane, so
#      `anatomical_roll_ref` falls back to the arbitrary edge-0 tangent), and the gun came out
#      lying sideways instead of along the limb.
#   2. those regions top out at -120 cm, and the Glock is 18 cm long while the ankle is only
#      ~10 cm off the floor -- so ANY seat centred on the ankle proper hangs the muzzle through
#      the sole. A real ankle rig sits with the muzzle at the ankle bone and the grip up the
#      lower shin, which is exactly this band: it puts the muzzle 6.7 cm above the floor with
#      no part of the gun below the sole.
# An OUTWARD azimuth (+90 = model's left, -90 = right) gives the seat a horizontal normal,
# which restores roll=0 -> "long axis along the limb" -> gun parallel to the leg.
ANKLE = (-1.16, -1.06)


def vertex_parts(model):
    """(10475,) int joint label per vertex, from the model's LBS weights."""
    W = model.lbs_weights.detach().cpu().numpy()      # (V, 55)
    return W.argmax(1)


def canonical_mesh(model):
    """Rest-pose, mean-shape verts + faces in SMPL-X's own frame (x=left, y=up,
    z=forward). Used ONLY to decide which faces belong to which site."""
    import torch
    with torch.no_grad():
        out = model()
    V = out.vertices[0].numpy().astype(np.float64)
    F = model.faces.astype(np.int64)
    return V, F


def face_regions(model):
    """dict region -> bool mask over faces. A face is in a region if a majority of
    its verts carry that region's joint labels."""
    vlab = vertex_parts(model)
    F = model.faces.astype(np.int64)
    out = {}
    for region, joints in REGIONS.items():
        jset = {NAME2J[j] for j in joints}
        vin = np.isin(vlab, list(jset))               # (V,) bool
        out[region] = vin[F].sum(1) >= 2              # >=2 of 3 verts in region
    return out


def _face_centroids_normals(V, F):
    tri = V[F]
    c = tri.mean(1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    return c, n


# A carry site = an anatomical region, a facing (which way the body surface points
# in the CANONICAL pose), and an optional height band. `facing` is in SMPL-X frame:
# +z is the direction the body faces, so front-facing surface has n_z > 0.
#   (region, facing, z_lo, z_hi)
# `facing` is either one of {"front","back","side_l","side_r","any"} or a FLOAT AZIMUTH IN
# DEGREES about the body's vertical axis, measured from +z (front) toward +x (the model's
# LEFT): 0 = front, +90 = model's left, 180 = back, -90 = model's right. The float form is
# what makes the 45-deg belt diagonals expressible; the strings are exact aliases of the
# four cardinal azimuths and are kept so existing sites behave identically.
SITES = {
    "appendix":      ("waistband", "front", None, None),   # front waistband, classic IWB
    "belly":         ("abdomen",   "front", None, None),
    "chest":         ("chest",     "front", None, None),
    "small_of_back": ("lower_back", "back", None, None),
    "upper_back":    ("upper_back", "back", None, None),
    "thigh_r":       ("right_thigh", "front", None, None),
    "thigh_l":       ("left_thigh",  "front", None, None),
    "shin_r":        ("right_shin",  "front", None, None),
    "shin_l":        ("left_shin",   "front", None, None),
    # ankle holster: INSIDE of the leg, gun parallel to the limb, muzzle at the ankle bone and
    # grip up the lower shin -- see the ANKLE note above for why this rides the SHIN region.
    # Inboard means the seat normal points ACROSS the body: the right leg's inner face looks
    # toward +x (the model's left) and vice versa.
    "ankle_r":       ("right_shin", +90.0, *ANKLE),
    "ankle_l":       ("left_shin",  -90.0, *ANKLE),
    "flank_r":       ("abdomen",   "side_r", None, None),   # side of the midriff
    "flank_l":       ("abdomen",   "side_l", None, None),
    "upper_arm_r":   ("right_upper_arm", "any", None, None),
    "forearm_l":     ("left_forearm",    "any", None, None),

    # --- the belt ring: 8 carry azimuths at 45 deg spacing, one height band ------------
    # Same region + same height band for all eight, so a sweep around the ring varies ONLY
    # the azimuth -- which is the whole point of the stop-and-pose experiment.
    "wb_front_mid":   ("beltline",    0.0, *BELT),
    "wb_front_left":  ("beltline",  +45.0, *BELT),
    "wb_left":        ("beltline",  +90.0, *BELT),
    "wb_back_left":   ("beltline", +135.0, *BELT),
    "wb_back_mid":    ("beltline",  180.0, *BELT),
    "wb_back_right":  ("beltline", -135.0, *BELT),
    "wb_right":       ("beltline",  -90.0, *BELT),
    "wb_front_right": ("beltline",  -45.0, *BELT),
}

_FACING_AXIS = {  # (axis, sign) in SMPL-X frame x=left, y=up, z=forward
    "front":  (2, +1),
    "back":   (2, -1),
    "side_l": (0, +1),
    "side_r": (0, -1),
}


def site_faces(model, site, *, facing_thresh=0.35, az_tol=30.0, min_horiz=0.30,
               min_faces=8):
    """Candidate face indices for a named carry site, on the canonical mesh.

    az_tol / min_horiz apply only to azimuth (float) facings: a face qualifies if its normal
    points within `az_tol` degrees of the target azimuth IN PLAN VIEW, and is not so close to
    horizontal (facing straight up/down) that its azimuth is meaningless. Selecting on the
    plan-view angle rather than a raw dot product matters on the belt, where the surface tilts
    noticeably in y -- a dot-product threshold would silently reject the tilted faces and bias
    each ring seat toward whichever part of its band happens to be most vertical.
    """
    if site not in SITES:
        raise KeyError(f"unknown site {site!r}; have {sorted(SITES)}")
    region, facing, z_lo, z_hi = SITES[site]
    V, F = canonical_mesh(model)
    mask = face_regions(model)[region].copy()
    c, n = _face_centroids_normals(V, F)
    if isinstance(facing, str):
        if facing != "any":
            ax, sg = _FACING_AXIS[facing]
            mask &= (sg * n[:, ax]) > facing_thresh
    else:                                             # azimuth in degrees
        horiz = np.hypot(n[:, 0], n[:, 2])
        az = np.degrees(np.arctan2(n[:, 0], n[:, 2]))
        daz = np.abs((az - float(facing) + 180.0) % 360.0 - 180.0)
        mask &= (horiz > min_horiz) & (daz <= az_tol)
    if z_lo is not None:
        mask &= c[:, 1] >= z_lo                       # SMPL-X y = up
    if z_hi is not None:
        mask &= c[:, 1] <= z_hi
    idx = np.flatnonzero(mask)
    if len(idx) < min_faces:
        raise ValueError(f"site {site!r} matched only {len(idx)} faces "
                         f"(region={region}, facing={facing}) -- loosen the filter")
    return idx


def sample_placement(model, site, rng, *, jitter=True):
    """Draw one placement on a site -> dict(face, bary, site). Uniform over the
    site's faces (area-weighted), barycentric inside the face. With jitter=False
    it returns the site's area-centroid face, for reproducible 'canonical' spots."""
    idx = site_faces(model, site)
    V, F = canonical_mesh(model)
    tri = V[F[idx]]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                         tri[:, 2] - tri[:, 0]), axis=1)
    if not jitter:
        c = tri.mean(1)
        hub = c.mean(0)
        f = idx[np.argmin(np.linalg.norm(c - hub, axis=1))]
        b = np.array([1 / 3, 1 / 3, 1 / 3])
    else:
        f = int(rng.choice(idx, p=area / area.sum()))
        r1, r2 = rng.random(), rng.random()
        su = np.sqrt(r1)
        b = np.array([1 - su, su * (1 - r2), su * r2])
    return dict(site=site, face=int(f), bary=b.tolist())


if __name__ == "__main__":
    import smplx
    MP = "/Users/jonah/Documents/code/model_gen/smplx_models"
    m = smplx.create(MP, model_type="smplx", gender="neutral", use_pca=False,
                     flat_hand_mean=True, batch_size=1)
    V, F = canonical_mesh(m)
    print(f"canonical mesh: {len(V):,} verts, {len(F):,} faces")
    print(f"{'site':<15}{'faces':>7}  {'centroid (x,y,z) cm in SMPL-X frame':>20}")
    for s in SITES:
        try:
            idx = site_faces(m, s)
            c = V[F[idx]].mean(1).mean(0) * 100
            print(f"{s:<15}{len(idx):>7}   ({c[0]:+6.1f},{c[1]:+6.1f},{c[2]:+6.1f})")
        except ValueError as e:
            print(f"{s:<15}   FAIL  {e}")
