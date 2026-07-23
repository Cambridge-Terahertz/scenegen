# SMPL-X body generation (scatterer-cloud source)

The all-Python chain that turns a licensed SMPL-X body model into a λ/8 THz
scatterer cloud for the forward model. Consolidated here (2026-07-23) from three
untracked, byte-identical copies that lived in driver dirs
(`scratch_results/{threat_dataset,smplx,stop_pose}`, `~/{threat_dataset,walkthrough}`).

## Files (tracked)

| file | role | key entry points |
|---|---|---|
| `bodies_smplx.py` | **the generator** — posed SMPL-X mesh → 0.5λ subdivide → λ/8 sample | `smplx_scene()` → `(q, s, n)` |
| `mesh_sample.py` | low-discrepancy (Van der Corput) surface sampler, exact mesh normals, front-cull | `sample_mesh()`, `median_nn()` |
| `bodylib.py` | alt SMPL-X wrapper used by threat_dataset/walkthrough drivers (was `body.py`) | `get_model()`, `posed_mesh()`, `to_sim_frame()` |
| `stance.py` | canned poses (e.g. `screening_pose()`) — body_pose to feed the generator | |
| `anatomy.py` | SMPL-X joint name↔index map (`NAME2J`), used by `stance.py` | |

Chain: `smplx_scene()` calls `smplx.create` → poses with betas/body_pose →
maps to the sim frame → `subdivide_front` (0.5λ facets, anti-alias) →
`sample_mesh` at **λ/8** NN spacing (`target_nn_frac=0.125`). Sampling is uniform,
so `s` is a **pure isotropic material scalar — NO density reweighting needed**
(that's the whole point; `bodies.py::density_weights` is only the fallback for
non-uniform clouds). Specular directionality is emergent from the coherent
forward sum, which is why λ/8 (not λ/2) is required.

## Assets — referenced by location, NEVER committed

- **SMPL-X models** (`SMPLX_{NEUTRAL,MALE,FEMALE}.npz`, v1.1, 104 MB each,
  Meshcapade non-commercial license): set env var **`SMPLX_MODELS`** to the dir
  containing `smplx/SMPLX_*.npz`. Mac → `model_gen/smplx_models/`; servers →
  `~/smplx_models/`. The dir is gitignored (`9070c70`). Use the 22-key v1.1
  files (with hand components); the stripped 11-key files are unloadable.
- **Gun mesh**: `~/walkthrough/glock17_ext.stl` — sample with `sample_mesh` at the
  same λ/8 spacing.
- **Generated clouds** (~40M pts/body): reproducible from (code + model + betas +
  pose) — regenerate on demand, do not store.
- `AMASS_ROOT` (`~/amass`): motion data, barely used; sims use `stance.screening_pose()`.

## Note

`bodylib.py` and `bodies_smplx.py` are two parallel SMPL-X wrappers (both call
`smplx.create`). The generator path for the forward model is `bodies_smplx.smplx_scene`.
Drivers still importing the old scattered `body.py`/`bodies_smplx.py` should be
migrated to import these tracked copies.
