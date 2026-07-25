# dataset4 detection analysis (plan §7 step 5)

Same feature *methodology* as the dataset2/3 probe (blob shape + range extent, body-grouped CV,
shuffle control), but the **experimental design changed** because dataset4's negatives are benign
*objects*, not bare bodies: **site-matched is the primary metric; pooled is only a leak measurement.**

**Results figure: [`fig7_detect_results.png`](fig7_detect_results.png)** (6 panels — group AUC, ROC,
per-confuser false-alarm, and per-object contrast / range-footprint / image-area distributions).

Scripts: `ds4_features.py` (per-scan features) → `ds4_feat.csv` (3,920 rows, 49 cols) → `ds4_detect.py`
(tables) / `ds4_plots.py` (the figure).
Feature tiers, kept strictly separate so leakage can't sneak in:
- **BLIND / deployable** — `han_db` brightness, blind blob shape, and range structure from `field_gated`
  (the ±5 cm shell that `han_db.max(1)` discards). Computable at test time; has a real-data analog (`ffts`).
- **PROBE / label-located** — object-window image shape + range (uses the known box; eval only).
- **CEILING / truth** — `n_obj_vox`, contrast (label-derived; upper bound, not a feature).

Site-matched subset: **882 threat / 1,862 benign** over 9 shared sites (belly, thigh_l/r, wb_front/back_×3).

## Headline numbers

Single-feature rank-AUC (threat vs benign):

| feature | family | AUC pooled | AUC site-matched |
|---|---|---|---|
| contrast_db | brightness (truth-ish) | 0.801 | **0.641** |
| hb_max / hb_p99 | brightness (blind) | 0.50 | 0.50 |
| ow_area (image area @ object) | image-2D shape | 0.44 | **0.39** |
| ow_asp (image aspect @ object) | image-2D shape | 0.59 | 0.50 |
| bl10_maxarea / maxasp | blind 2D blob | 0.49 | 0.49 |
| rng_p99spread (thickness) | range-3D | 0.54 | 0.71 |
| **rng_blob_maxarea (off-surface footprint)** | **range-3D** | 0.62 | **0.891** |
| n_obj_vox (truth size) | ceiling | 0.999 | **1.000** |

GroupKFold-over-98-bodies AUC, **site-matched** (mean±std; shuffle control in last col):

| feature group | pooled | fold mean | shuffle |
|---|---|---|---|
| BRIGHT | 0.537 | 0.536±0.03 | 0.528 |
| SHAPE2D (blind blobs) | 0.507 | 0.507±0.01 | 0.473 |
| **RANGE3D (blind)** | **0.960** | 0.960±0.005 | 0.518 |
| ALL_BLIND | 0.958 | 0.958±0.007 | 0.517 |
| PROBE_shape | 0.687 | 0.689±0.01 | 0.493 |
| PROBE_range | 0.913 | 0.914±0.006 | 0.471 |
| CEILING | 1.000 | 1.000 | 0.504 |

Per-confuser false-alarm @ Pd=0.80 (site-matched OOF): **ALL_BLIND** buckle 0.07, phone 0.08, badge 0.00,
wallet 0.01 (all-benign 0.04). The two metals (buckle, phone) are the worst false alarms, as expected.

### "blind" ≠ transfer-honest — decomposing the 0.96

`ALL_BLIND` = **label-blind** (no object location; shuffle control 0.52), but it is **not leakage-free**:
its `field_gated` features are gated on `depth_map` (a leakage-listed field), and its power is the RF
encoding of the "object sits proud in the clothing gap" cue — the same signal as the `depth_map` bump.
Stripping that depth-anchored cue:

| tier (site-matched GroupKFold AUC) | AUC |
|---|---|
| brightness only | 0.54 |
| + 2D image shape | 0.53 |
| **+ range *thickness* (no depth reference) = transfer-honest blind** | **0.73** |
| + gap cue (depth-anchored off-surface footprint) → full `ALL_BLIND` | 0.96 |
| gap cue *alone* | 0.944 |
| truth ceiling (`n_obj_vox`) | 1.00 |

So **the 0.96 is ~entirely the fragile gap cue** (0.944 by itself). A deployment-honest blind detector
is **~0.73**, carried by range thickness (0.70 single-feature), which itself still needs the
roughness/clothing test. Without range at all, it is chance (0.53). Report 0.73 as the sober blind
estimate; treat 0.96 as gap-cue-inflated.

## What it means

1. **Brightness does not discriminate at matched sites** (0.64). The steel buckle *beats* the gun on
   contrast (gun-vs-buckle AUC 0.31) and the alloy phone matches it (0.60) — both metals.
2. **Image-2D shape is WORSE than chance** (ow_area 0.39). In the specular sim the gun images as a
   compact glint while flat benign slabs mirror their whole face → the confusers look *bigger and more
   elongated*. So the "shape rescues brightness" hope from dataset3 does **not** hold against flat
   metal/dielectric confusers. This is new and only visible with real confuser negatives.
3. **The blind 0.96 is almost entirely a transfer-fragile cue; the honest blind number is ~0.73.**
   "blind" = label-blind, NOT leakage-free (see decomposition above). `rng_blob_maxarea` works because
   every object sits ~1.2 cm proud in the clothing gap, so its return peaks *off* the body surface; the
   largest off-surface region is the object (12/12 spatially on the object) and the gun has the biggest
   footprint. That gap cue alone scores 0.944; **it is the RF twin of the `depth_map` 1.2 cm bump the plan
   flagged as non-transferable — clothing fills the gap on real data.** Strip it and the transfer-honest
   blind detector is **0.73**, carried by range *thickness* (0.70), which still needs the roughness test.
4. **`n_obj_vox` = 1.000** confirms the discriminating information is fully present in object geometry;
   the deployable question is whether a *transfer-robust* feature recovers it. In the specular sim the
   only strong recovery is the fragile gap cue.

## Pooled vs site-matched (the leak, and a reversal)

| group | pooled (all sites) | site-matched | Δ |
|---|---|---|---|
| ALL_BLIND | 0.808 | 0.958 | **−0.150** |
| CEILING | 0.999 | 1.000 | −0.001 |

The blind pooled AUC is *lower* than site-matched (unusual). Two compounding optimisms in the
site-matched number: (a) it drops the 784 threat-only flank/ankle scans, where the grazing gun presents
a small off-surface footprint (median blob 53 vs 148 broadside) and is genuinely hard; (b) the strong
feature is the fragile gap cue. Report both; treat neither as a sim→real headline.

## Bottom line for the strategic question

Reuse the pipeline and features from the 960-scan set — **but not its pooled-AUC framing.** dataset4's
value is exactly that it breaks the dataset3 story: with real confuser negatives, brightness overlaps,
image-shape inverts, and the only strong blind signal is a gap cue that won't survive clothing. This is
the concrete argument for **§7 step 2 (roughness + noise floor + clothing) as the true next milestone** —
it is the only way to tell whether *any* of this survives, and this pipeline can be re-run on the
roughened set to measure the drop directly.

## Caveat

Specular sim (no roughness/noise/clothing). Benign slab magnitudes overstated; the range gap cue is
optimistic. Read every number as an upper bound, not a sim→real prediction.
