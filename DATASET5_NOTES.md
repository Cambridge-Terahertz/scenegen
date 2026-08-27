# dataset5 — expanded threat + benign object set (launched 2026-07-24)

Same physics and save format as dataset4 (the 3,920-scan set), with a much larger, curated
object catalogue and a 45-site placement space. Purpose: a richer object-discrimination set —
more threat variety (5 guns + 5 knife/tool types) and more benign confusers — so the detector's
decision boundary is forced onto object identity across many carry positions.

## Run at a glance
| | |
|---|---|
| bodies | **200** (2 gender × 10 height × 10 build), `shapes_10x10.json` (SHAPES_LEVELS=10 gen_shapes7.py) |
| placements/body | **415** (360 threat + 53 benign + 2 baseline) |
| total scans | **83,000** (72,000 threat / 11,000 no-threat) |
| host / GPU | tesla, RTX 5090 (idle at launch) |
| output | `tesla:~/dataset5_2026-07-24/` (~80 GB) — **NOT on NAS yet; rsync when done** |
| driver | `~/probe/dataset5_driver.py` (tesla) / `stop_pose/dataset5_driver.py` (mac) |
| launched | 2026-07-24 21:33, detached (`setsid`), log `~/probe/ds5.log`; resume-safe (skips existing npz) |
| est. runtime | ~5 days (~30–40 min/body) |

## Objects (17) — `threat_dataset/objects_v5/` + `manifest.json`
Built by `process_objects.py`: each mesh canonicalized (long=x, mid=y, thin=z), scaled to a real
size, tagged a material + a seat-roll offset.

**Threats (10):**
- `gun1_glock17` — the SAME two-material Glock 17 as dataset4/3 (steel slide + polymer frame via
  `threats._glock_parts`). **The cross-set anchor** — reused deliberately for comparison.
- `gun2_glock19` (Glock 19), `gun3_hellcat` (FMK 9C1 ≈ Springfield Hellcat), `gun4_mp9`
  (Grand Power K100 ≈ S&W M&P9), `gun5_derringer` (Tanfoglio GT27 ≈ pocket pistol). Single steel.
- `knife_switchblade` / `knife_folding` / `knife_corkscrew` — 3 closed folding pocket-tools, baked
  from the rigged Pocket Knifes A FBX (posed to the closed animation frame). `knife_hunting` — fixed
  ~6" blade. `knife_dielectric` — the folding knife with ceramic material (s=0.35).

**Benign confusers (7):** `phone_nokia`, `phone_modern` (iPhone-16 slab), `wallet`, `badge`
(name badge), `keys` (3D splayed bunch), `necklace` (pendant), `belt` (buckle).

Materials: gun1 two-material; all other guns + knives + keys/necklace/belt = steel (s=1.0);
phones = alloy (0.95); wallet = plastic (0.30); badge = polymer (0.35); dielectric knife = ceramic.
(User decision 2026-07-24: "rest steel" — the bulk-pack guns' .mtl is cosmetic paint, not a physical
steel/polymer split, so no per-gun two-material was possible without a height-split.)

## Placement — `threat_dataset/placement_config.py`
Registers a 40-site catalog into `anatomy.SITES` (reconciles the confusers' reviewed 26-site set
with the threats' 45→36-leaf set: waistband ring, flank, midline, thigh front/back/inner/outer,
ankle inside/outside, shin front/back/inner/outer, arms). Object→site map:
- **threats → all 36 sites** (user: threats go everywhere).
- **confusers → their reviewed OK sites** (`confuser_site_map.json`, from the placement-review
  Artifact; maybe-sites excluded).

Each object seats with its own roll offset (manifest). `stance_of()` splits sites front/back so the
body is turned appropriately (9 back sites → back stance, 27 → front).

## Save format — identical to dataset4 (comparable)
Per npz: `han_db` (2D dB max-projection), **`field_gated`** (±5 cm complex range-gated shell,
`(n_cells, K=21)` complex64) + **`gate_cells`** (the (ix,iz) lookup; iy implicit from `depth_map`),
`depth_map`, `obj_vox`, `obj_aabb`, `seat_xyz`, `visibility`, `han_contrast_db`, betas/site/stance,
gun_* aliases on threats, and raw `sig` every 128th scan. Standoff 2.0 m, taylor(sll=40) both tapers,
1 cm cross-range grid, 5 mm range step, phase-recursion on.

## Provenance / how to regenerate
`DS5_OUT=~/dataset5_2026-07-24 ~/miniconda3/envs/ra/bin/python dataset5_driver.py` on tesla
(from `~/probe`; geometry modules resolve from `~/walkthrough`, forward model from
`~/Documents/code/model_gen/python`). `DS5_NBODIES=N` truncates; `DS5_SHAPES=path` overrides shapes.

## Caveats
- **SPECULAR** — no roughness, no noise floor, no clothing (plan §7 step 2, still deferred). Read
  contrast as relative ordering, an upper bound, not a sim→real prediction.
- First-scan sanity (body 1): rigid-object visibility ~0.4–0.45 (correct; NOT the conformal 1.000
  double-count), contrast +17–30 dB. Looks right.

## TODO
- When generation completes: **rsync `tesla:~/dataset5_2026-07-24` → NAS** `Jonah/synthetic_data/`.
- Then run the detection analysis (reuse the dataset4 pipeline: features → GroupKFold → site-matched).

## PERFORMANCE TODO — port two validated speedups from the walkthrough dataset (2026-07-30)

Both were measured on `walkthrough_dataset_driver.py` (moving-portal dataset, same dataset5 delta-forward
physics) and produce **bit-identical output** (max|delta|/peak = 0.00e+00). dataset5 does the same full-grid
recon + full-cloud per-placement occlusion on 83k scans, so both should give a large end-to-end speedup.
Reference impl: `~/Documents/scratch_results/smplx/walkthrough_dataset_driver.py`.

1. **Shell-only reconstruction via `recon.backproject_points`.** Today `recon_complex()` backprojects the
   full dense grid (181 x Ny x 211 ~ 4M voxels) and `gate()` then keeps only the +/-5 cm surface shell
   (~100k pts), discarding ~97%. Instead, compute the shell eval points from `depth_map` (for each finite
   cell ix,iz: KGATE range samples about the surface) and call `backproject_points(sig*W, ..., points=shell)`
   -- returns the identical `field_gated`/`gate_cells` (test-verified vs grid). `han_db` becomes the per-cell
   shell max (matches the full max-projection within ~0.5 dB). ~40x cheaper backproject.

2. **Localized occlusion.** Today each placement does `q_all = vstack([bq ~40M body pts, gq]); zbuffer_cull(q_all)`
   -- re-culls the WHOLE body cloud per object; on the walkthrough this was 65% of per-placement time.
   Instead cull only body points inside the object's cross-range footprint (bbox of `gq` + ~1 cm margin,
   AND-ed with the body-only `keep_b0`): `near = keep_b0 & in_footprint; zbuffer_cull(vstack([bq[near], gq]))`.
   EXACT -- a body point outside the object footprint shares no zbuffer cell with any object point (so its
   visibility is unchanged), and no far body point can shadow the object. Cut occlusion 278s to 34s (8x) for
   36 placements, bit-identical.

Net on the walkthrough: per-placement 20 s to 5 s (4x). A third idea -- folding the shadow-subtract +
object-add into one `fwd([shadow;object],[-skin;s_obj])` call -- is correct but gave ~0 speedup there since
the forward was not the bottleneck; low priority.

---

## Coordinate frame (adopted 2026-08-26) — read this before touching any driver

**`+z` into the scene with z = 0 at the array · `+y` up · `+x` to the LEFT · right-handed.**
Stated once in `frame.py`; matches `reflect_array_imaging/processing/bistatic_recon.py`, so a
simulated scan and a DK3 scan share one frame. The legacy sim frame was (x = array-right,
y = RANGE, z = vertical); `frame.legacy_to_world` converts, determinant +1.

**Plot rule:** always draw +x leftward (`ax.invert_xaxis()`). Negating x and inverting the axis
cancel, so figures look unchanged — what changes is the axis sign and labels. Applying only ONE of
the two silently mirrors the figure, so every `set_xlim` must pass reversed bounds.

**Reading old data:** ds3–ds5b and the ds6 pilot are legacy-frame. They stay readable — each npz
carries its own `xs/ys/zs` — but world-frame code will silently mis-index them. **Always read via
`frame.load_scan()`**, or call `frame.assert_frame()`.

**model_gen is frame-agnostic; the PIPELINE IS NOT.** `backproject` takes a point set and only forms
Euclidean distances (permuting the frame moves fp64 results by 5.99e-08). But three things outside
model_gen are axis-aware and must be given the right axis, or they silently corrupt every scan
(measured: 20% fewer visible points, 20° obliquity error, contrast inflated 5 dB):

| call | legacy default | world value |
|---|---|---|
| `bodies.zbuffer_cull(q, …, depth_axis=1)` | axis 1 | **`depth_axis=2`** |
| `attach.front_cull(…, view_dir=(0,-1,0))` | −y | **`(0, 0, -1)`** |
| `obliquity_deg` reads `gn[:,1]` | col 1 | **`gn[:,2]`** |

**Validating a frame change:** never by bit-comparison in fp32. This is interferometry at λ≈1 mm
over r≈2 m (~2000 wavelengths), so fp32 phase round-off moves results ~1e-3 relative (0.0088 dB)
purely from reordering `dx²+dy²+dz²`. Use fp64, or a 1e-3 tolerance. Reference tests:
`analysis_ds6/phase0*.py` (invariance), `parity.py` (driver), `loadertest.py` (loader),
`p3test.py` (packers).

**AWS:** the ECR image BAKES the driver, and `aws_ds6/build.sh` must copy `frame.py` next to it
(the driver imports it by path). `submit.sh` now REFUSES to submit when the driver is newer than
the pushed image — override only with `FORCE_STALE=1`.
