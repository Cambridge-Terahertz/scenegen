"""Posed SMPL-X body from an AMASS motion, in the SMPL-X frame.

AMASS gives us BOTH halves of the body-variation problem for free:
  * pose_body (63,) axis-angle per frame  -> the pose library (2079 CMU seqs)
  * betas (16,) per subject               -> an EMPIRICAL shape distribution over
    96 real people. Sampling (pose, betas) jointly from AMASS preserves the real
    correlation between build and gait, which independent N(0,1.3) betas destroy.
"""
from __future__ import annotations
import glob
import os
import numpy as np
import trimesh

# Mac defaults; bose keeps them in ~/smplx_models and ~/amass. Override with env vars
# so the same module runs on both without edits.
MP_DEFAULT = os.environ.get(
    "SMPLX_MODELS", "/Users/jonah/Documents/code/model_gen/smplx_models")
AMASS_ROOT = os.environ.get("AMASS_ROOT", "/Users/jonah/Downloads/amass")

_MODEL_CACHE = {}


def get_model(gender="neutral", model_path=MP_DEFAULT):
    key = (gender, model_path)
    if key not in _MODEL_CACHE:
        import smplx
        _MODEL_CACHE[key] = smplx.create(model_path, model_type="smplx",
                                         gender=gender, use_pca=False,
                                         flat_hand_mean=True, batch_size=1)
    return _MODEL_CACHE[key]


def amass_sequences(root=AMASS_ROOT):
    return sorted(glob.glob(f"{root}/*/*/*.npz"))


def load_amass(path):
    d = np.load(path, allow_pickle=True)
    return dict(pose_body=d["pose_body"].astype(np.float64),
                betas=d["betas"][:10].astype(np.float64),
                gender=str(d["gender"]),
                trans=d["trans"].astype(np.float64),
                root_orient=d["root_orient"].astype(np.float64),
                fps=float(d["mocap_frame_rate"]))


def posed_verts(model, betas=None, body_pose=None, global_orient=None):
    """-> (V (10475,3) float64, F (20908,3) int64) in the SMPL-X frame
    (x=left, y=up, z=forward). Height-centered on the vertex bbox."""
    import torch
    kw = {}
    if betas is not None:
        b = np.zeros(model.num_betas, np.float32)
        bv = np.asarray(betas, np.float32).ravel()
        b[:min(len(bv), model.num_betas)] = bv[:model.num_betas]
        kw["betas"] = torch.as_tensor(b[None])
    if body_pose is not None:
        kw["body_pose"] = torch.as_tensor(
            np.asarray(body_pose, np.float32).reshape(1, -1))
    if global_orient is not None:
        kw["global_orient"] = torch.as_tensor(
            np.asarray(global_orient, np.float32).reshape(1, 3))
    with torch.no_grad():
        out = model(**kw)
    V = out.vertices[0].numpy().astype(np.float64)
    F = model.faces.astype(np.int64)
    return V, F


def canonical_verts(model):
    """Rest-pose mean-shape verts -- the reference the roll transport is measured on."""
    return posed_verts(model)[0]


def posed_mesh(model, **kw):
    """Posed trimesh with consistent outward normals (+ per-vertex normals for the
    conformal attach)."""
    V, F = posed_verts(model, **kw)
    m = trimesh.Trimesh(vertices=V, faces=F, process=False)
    m.fix_normals()
    return m


# ---- world frames: apply to body AND threat together, after attachment ---------

def to_sim_frame(P, standoff=2.0, z_center=None):
    """SMPL-X (x=left,y=up,z=fwd) -> single-sensor sim frame
    (x=width, y=depth from the array, z=height). Rotation, det +1."""
    x = P[:, 0]
    z = P[:, 1] if z_center is None else P[:, 1] - z_center
    y = standoff - P[:, 2]
    return np.column_stack([x, y, z])


def to_portal_frame(P, walk_y=0.0, z_center=None):
    """SMPL-X -> portal frame (x=width, y=walk direction the person faces, z=height)."""
    x = P[:, 0]
    y = P[:, 2] + walk_y
    z = P[:, 1] if z_center is None else P[:, 1] - z_center
    return np.column_stack([x, y, z])
