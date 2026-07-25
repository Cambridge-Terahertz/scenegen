# scenegen — synthetic concealed-threat screening dataset pipeline

Builds labeled synthetic mmW/THz body-screening datasets: poses SMPL-X bodies, seats threat +
benign objects at carry positions, then runs the RF forward model + backprojection (model_gen) to
produce labeled reconstructed scans (2D `han_db` + the ±5 cm 3D range-gated shell). Sits on top of
`model_gen` (the RF forward model / recon).

## Layout
- **root** — scene/threat modules (`threats`, `anatomy`, `attach`, `mesh_sample`, `body`, `stance`);
  dataset drivers (`dataset{3,4,5}_driver.py`); `placement_config.py` (object x site map);
  `process_objects.py` (mesh -> sim-ready `objects_v5`); shape generators (`gen_shapes7.py`,
  `shapes_*.json`); `objects_v4/` + `objects_v5/manifest.json`; `DATASET5_NOTES.md`.
- **objprep/** — object prep + the placement-review gallery (`confuser_gallery.py`, keys/fold tools).
- **analysis/** — detection analysis (`ds4_features/detect/plots`, `ds3_detect`).

## Running (tesla, GPU + CuPy in the `ra` env)
`DS5_OUT=~/dataset5_YYYY-MM-DD ~/miniconda3/envs/ra/bin/python dataset5_driver.py`

External deps (NOT in repo): `model_gen` (`~/Documents/code/model_gen/python` — RF forward model /
recon, its own repo) and the SMPL-X models (`~/smplx_models`). Outputs (`*.npz`), the large
`objects_v5/*.obj` meshes, logs, and render PNGs are gitignored.

Canonical modules were consolidated from `~/walkthrough` (the copies the drivers actually import;
`anatomy.py`/`attach.py` had drifted from other copies). See `DATASET5_NOTES.md` for the live run.
