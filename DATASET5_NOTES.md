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
