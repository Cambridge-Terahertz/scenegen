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
HIP_OUT_DEG = 7.0      # shoulder-width feet


def _idx(joint_name):
    """Index into pose_body (63,) for a named body joint."""
    return NAME2J[joint_name] - 1


def screening_pose():
    """-> body_pose (63,) float64. Arms lowered from the T-pose, feet shoulder-width."""
    p = np.zeros((21, 3), np.float64)
    a = np.deg2rad(ARM_DOWN_DEG)
    p[_idx("left_shoulder")] = [0.0, 0.0, -a]      # left arm rests along +x -> -z lowers it
    p[_idx("right_shoulder")] = [0.0, 0.0, +a]     # right arm rests along -x -> +z lowers it
    h = np.deg2rad(HIP_OUT_DEG)
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
