"""SMPL-X body -> THz forward-model scatterer cloud.

Generates a posed SMPL-X mesh (smplx package), maps it to the sim frame, Loop-
subdivides the front to <= lambda/2 facets (so the smooth body isn't facet-aliased),
and low-discrepancy samples it via mesh_sample.sample_mesh -> (q, s, n).

Reflectivity is a per-point ISOTROPIC scalar `s` (material only, NO angle factor —
specular directionality is emergent from the coherent forward sum). Uniform sampling
means no density weighting is needed, so `s` is pure material. A per-part material map
can be supplied (vertex-label -> s) using SMPL-X's fixed topology.

Frame: SMPL-X is y-up, +z forward (facing). Sim frame: x=width, y=depth (array at
y=0 looking +y, body front toward small y at standoff), z=height. Mapping
(x,y,z)_smplx -> (x, standoff - z, y)_sim is a rotation (det +1, winding preserved).
"""
from __future__ import annotations
import numpy as np
import trimesh

from mesh_sample import sample_mesh, median_nn

C0 = 299_792_458.0


def smplx_to_sim(V, standoff=2.0, z_center=True):
    """Map SMPL-X (x=left, y=up, z=forward) verts -> sim frame; center height."""
    x = V[:, 0]
    z = V[:, 1]                                   # up -> height
    if z_center:
        z = z - z.mean()
    y = standoff - V[:, 2]                        # forward(+z) -> nearest array (small y)
    return np.column_stack([x, y, z])


def smplx_mesh(model_path, *, gender="neutral", betas=None, body_pose=None,
               global_orient=None, standoff=2.0, use_pca=False):
    """Build a posed SMPL-X mesh and return a trimesh.Trimesh in the sim frame.

    model_path : dir containing the licensed SMPLX_{NEUTRAL,MALE,FEMALE}.npz files
                 (the smplx package looks for <model_path>/smplx/SMPLX_<GENDER>.npz).
    betas      : (10,) or (n_betas,) shape coeffs (None -> mean shape).
    body_pose  : (21,3) or (63,) axis-angle joint rotations (None -> rest pose).
                 Source realistic poses from AMASS.
    """
    import torch, smplx
    model = smplx.create(model_path, model_type="smplx", gender=gender,
                         use_pca=use_pca, flat_hand_mean=True, batch_size=1)
    kw = {}
    if betas is not None:
        b = np.zeros(model.num_betas, np.float32)            # pad/trim to model's num_betas
        bv = np.asarray(betas, np.float32).ravel()
        b[:min(len(bv), model.num_betas)] = bv[:model.num_betas]
        kw["betas"] = torch.as_tensor(b[None])
    if body_pose is not None:
        bp = np.asarray(body_pose, np.float32).reshape(1, -1)
        kw["body_pose"] = torch.as_tensor(bp)
    if global_orient is not None:
        kw["global_orient"] = torch.as_tensor(np.asarray(global_orient, np.float32)[None])
    with torch.no_grad():
        out = model(**kw)
    verts = out.vertices[0].numpy().astype(np.float64)
    faces = model.faces.astype(np.int64)
    verts = smplx_to_sim(verts, standoff)
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    m.fix_normals()                              # consistent outward winding
    return m


def subdivide_front(mesh, target_edge, *, margin=0.25):
    """Keep front-facing faces (n_y < margin, generous to include silhouette),
    Loop-subdivide the FULL watertight mesh until median edge <= target_edge, then
    let the sampler front-cull. Subdividing the whole closed mesh (not a front-culled
    open patch) avoids non-manifold boundary edges through the hands/fingers, which
    Loop subdivision otherwise blows up into runaway vertices. `margin` is unused
    (kept for signature compat). Returns (verts, faces, iters)."""
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    med = np.median(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1))
    iters = max(0, int(np.ceil(np.log2(med / target_edge))))
    if iters:
        V, F = trimesh.remesh.subdivide_loop(V, F, iterations=iters)  # smooth limit, watertight
    return np.asarray(V, np.float64), np.asarray(F, np.int64), iters


def smplx_scene(model_path, *, target_nn_frac=0.125, f_center=290e9, standoff=2.0,
                gender="neutral", betas=None, body_pose=None, s_body=0.7,
                material_map=None, facet_frac=0.5):
    """Full body -> (q, s, n). target_nn_frac and facet_frac are in wavelengths."""
    lam = C0 / f_center
    mesh = smplx_mesh(model_path, gender=gender, betas=betas, body_pose=body_pose,
                      standoff=standoff)
    V, F, iters = subdivide_front(mesh, facet_frac * lam)
    print(f"  subdivided front to {len(F):,} faces ({iters} Loop levels), "
          f"median edge {np.median(np.linalg.norm(V[F[:,0]]-V[F[:,1]],axis=1))/lam:.3f} lambda")
    q, n, info = sample_mesh(V, F, target_nn_frac * lam, front_cull=True)
    s = np.full(len(q), s_body, np.float64)       # per-point isotropic material scalar
    # (material_map hook: label each point by nearest body-part vertex -> s; TODO when needed)
    mm, frac = median_nn(q, lam)
    print(f"  sampled {len(q):,} pts, median NN {mm:.3f} mm = {frac:.3f} lambda "
          f"(target {target_nn_frac} lambda)")
    return q.astype(np.float32), s, n.astype(np.float32)


if __name__ == "__main__":
    import sys
    mp = sys.argv[1] if len(sys.argv) > 1 else None
    if mp is None:
        print("usage: bodies_smplx.py <model_path>  (dir with smplx/SMPLX_NEUTRAL.npz)")
        sys.exit(1)
    q, s, n = smplx_scene(mp, target_nn_frac=0.125)
    np.savez("/tmp/smplx_cloud.npz", q=q, n=n)
    print("saved /tmp/smplx_cloud.npz", q.shape)
