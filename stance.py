"""The 'stop and pose' screening stance, and the two viewing rotations.

SMPL-X's REST pose is a T-pose: the arms come straight out horizontally (measured on the
canonical mesh -- the `forearm_l` site's centroid sits at x=+54 cm, y=+4 cm, i.e. level with
the shoulder and half a metre out). So a screening stance is reached by rotating the arms DOWN
from horizontal, not up from the sides. Getting that backwards gives a body with its arms
folded across its chest, which would shadow exactly the waistband seats this experiment is
built to measure -- hence `render_seats.py` exists to eyeball it before any GPU time is spent.

Axis-angle conventions, SMPL-X frame (x = model's LEFT, y = up, z = forward/facing):
  * a rotation about +z swings a limb in the x-y (left/up) plane -- the abduction axis;
  * the left arm rests along +x, so a NEGATIVE z-rotation lowers it;
  * the right arm rests along -x, so a POSITIVE z-rotation lowers it;
  * the legs hang along -y, so the signs invert for the hips.

`pose_body` is 21 joints x 3, covering JOINT_NAMES[1:] -- so joint j lives at index j-1:
left_shoulder (16) -> 15, right_shoulder (17) -> 16, left_hip (1) -> 0, right_hip (2) -> 1.
"""
from __future__ import annotations
import numpy as np

from anatomy import NAME2J

ARM_DOWN_DEG = 40.0    # from horizontal. 40 deg down = arms ~50 deg off the torso: clear of
                       # the flanks, but not a comedic T-pose.
ARM_UP_DEG = 10.0      # L3/AIT portal "goalpost": upper arms held out nearly LEVEL with the
                       # ground, just clear of the shoulders ...
ELBOW_UP_DEG = 80.0    # ... and the elbows carry the rest, so the forearms still finish vertical
                       # (10 + 80 = 90). Shoulder and elbow turn about the same axis, so the two
                       # angles simply add -- keep their sum at 90.
WRIST_ROLL_DEG = 90.0  # ... and the forearms twisted so the PALMS FACE THE ARRAY. Without this
                       # the hands keep their T-pose palm-down orientation, which after the arm is
                       # raised leaves them edge-on: a hand seen edge-on presents almost no
                       # specular return, and is not the flat-palms-forward stance a portal asks for.
HIP_OUT_DEG = 7.0      # shoulder-width feet
HIP_OUT_L3_DEG = 12.0  # the portal's floor markings set the feet wider than a natural stance


def _idx(joint_name):
    """Index into pose_body (63,) for a named body joint."""
    return NAME2J[joint_name] - 1


def screening_pose(arms="down"):
    """-> body_pose (63,) float64. Feet apart; arm position selected by `arms`.

    "down" -- arms lowered from the T-pose. Datasets 3-6 were generated with this.
    "up"   -- the L3/AIT portal pose: upper arms raised above horizontal, elbows bent so the
              forearms finish vertical and the hands ride above the head. This is the stance a
              passenger actually holds in a millimetre-wave portal.

              It is a REALISM change, not a visibility one. Measured front-view occlusion over
              torso bands anchored in metres above the feet (anchoring by fraction of height is
              wrong here -- raising the arms grows the bbox 1.71 -> 1.98 m, so a fixed fraction
              lands on different anatomy): every band moves by <=3.3%, and whole-body visibility
              goes 85.7% -> 86.1%. At ARM_DOWN_DEG=40 the arms already hang clear of the torso,
              so there was little shadowing left to remove.

    The default stays "down" so the already-generated datasets reproduce bit-for-bit; a new run
    asks for "up" explicitly.
    """
    p = np.zeros((21, 3), np.float64)
    if arms == "down":
        a = np.deg2rad(ARM_DOWN_DEG)
        p[_idx("left_shoulder")] = [0.0, 0.0, -a]  # left arm rests along +x -> -z lowers it
        p[_idx("right_shoulder")] = [0.0, 0.0, +a] # right arm rests along -x -> +z lowers it
        h = np.deg2rad(HIP_OUT_DEG)
    elif arms == "up":
        a = np.deg2rad(ARM_UP_DEG)
        p[_idx("left_shoulder")] = [0.0, 0.0, +a]  # mirror of "down": +z RAISES the left arm
        p[_idx("right_shoulder")] = [0.0, 0.0, -a]
        # the elbow turns about the same +z, so the forearm keeps bending in the same (coronal)
        # plane the shoulder swung in -- shoulder + elbow = 90 deg leaves the forearms vertical.
        e = np.deg2rad(ELBOW_UP_DEG)
        p[_idx("left_elbow")] = [0.0, 0.0, +e]
        p[_idx("right_elbow")] = [0.0, 0.0, -e]
        # Pronation/supination: a twist about each forearm's OWN long axis, which in the rest pose
        # is +x for the left arm and -x for the right -- hence the opposite signs. Verified by
        # measuring the palm normal (PCA thin axis of the hand vertices), not by trusting the sign.
        r = np.deg2rad(WRIST_ROLL_DEG)
        p[_idx("left_wrist")] = [-r, 0.0, 0.0]
        p[_idx("right_wrist")] = [+r, 0.0, 0.0]
        h = np.deg2rad(HIP_OUT_L3_DEG)
    else:
        raise ValueError(f"arms must be 'down' or 'up', got {arms!r}")
    p[_idx("left_hip")] = [0.0, 0.0, +h]           # legs hang along -y, so the sign flips
    p[_idx("right_hip")] = [0.0, 0.0, -h]
    return p.ravel()


# The 180 deg turn is a VIEW, not a pose: same body_pose, rotated about the vertical (+y).
# Topology-anchored placements ride around with the body for free, so the seat is identical --
# only the sensor's line of sight to it changes.
VIEWS = {
    "front": np.array([0.0, 0.0, 0.0]),            # facing the sensor
    "back":  np.array([0.0, np.pi, 0.0]),          # turned away from it
}


def boresight_cos(normals_smplx, view):
    """cos of the angle between a seat's outward normal and the direction TOWARD the array.

    In the sim frame the array sits at y=0 looking +y with the body in front of it, so the
    direction from a target toward the array is -y. `body.to_sim_frame` maps SMPL-X
    (x, y, z) -> (x, standoff - z, y), so a SMPL-X normal (nx, ny, nz) becomes
    (nx, -nz, ny) and its dot with (0,-1,0) is simply +nz. Turning the body 180 deg about +y
    sends nz -> -nz. Hence:  front view -> +nz,  back view -> -nz.

    cos = 1 is a seat staring straight down the boresight; cos = 0 is grazing (invisible to a
    specular model); cos < 0 is facing away, i.e. self-occluded by the body.
    """
    nz = np.asarray(normals_smplx, np.float64).reshape(-1, 3)[:, 2]
    return nz if view == "front" else -nz
