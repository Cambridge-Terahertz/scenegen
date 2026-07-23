# Reflectarray THz simulator — Python/CuPy port

Port of the `model_gen` MATLAB+CUDA reflectarray imaging simulator to Python/CuPy.
Branch: **`python-port`**. Everything lives under `python/`.

## Status (2026-06-26)

The **physics pipeline is ported, GPU-accelerated, validated, and running at
production scale**: forward model → backprojection recon → layered result
storage → NAS → browse on the Mac.

| Component | File | Status |
|---|---|---|
| Forward model | `raforward.py` | ✅ fused CuPy `RawKernel`, fp32/fp64, NumPy fallback |
| Reconstruction | `recon.py` | ✅ time-domain backprojection (port of `bp.m`), fp32/fp64 |
| Result storage | `results.py` | ✅ `.h5` (chunked) + preview PNG + contact sheet → NAS |
| Reference/spec | `forward_model_ref.py` | ✅ float64 ground truth + float32 characterization |
| End-to-end driver | `run_model.py` | ✅ cloud → forward → recon → save, with timing |
| Cloud sources | `bodies.py` | ✅ `load_xyz` (real), `synthetic_body` (**fixture only**) |
| Benchmark | `bench.py` | ✅ dual-precision GPU timing |
| Tests | `*.py` (pytest) | ✅ 10+ pass (parity, PSF focus, float32 study) |

### Performance (RTX A4000, fp32)

| Scale | Forward | Recon | Total | vs MATLAB (~50 min) |
|---|---|---|---|---|
| 510k scatterers | 8.75 s | 3.9 s | ~13 s | — |
| **4M (run_008 scale)** | **71.6 s** | **2.8 s** | **~74 s** | **~40× faster** |

~123 G-eval/s sustained; scales linearly; 4M fits in the A4000's 16 GB.

## Architecture / data flow

```
cloud (q, s)  ──►  forward_model (raforward)  ──►  sig[nf, nrx]
   │                                                  │
   │  Scene(q, s, tx, rx, k)                           ▼
   └──────────────────────────────►  backproject (recon)  ──►  img[nx,ny,nz]
                                                          │
                                          save_run (results)  ──►  NAS
                                          results.h5 + meta.json + preview/*.png
```

- **Forward** (`raforward.forward`): bistatic scalar Born model,
  `sig[f,i] = Σ_t s_t/(r1·r2)·exp(-j k_f (r1+r2))`. One fused kernel over all
  `(freq, rx)` outputs; `r1` precomputed; distances/phase in the chosen precision,
  `__sincosf` + accumulation in fp32.
- **Recon** (`recon.backproject`): IFFT range-compression per receiver, then
  periodic (mod 1/df) delay-and-sum with `exp(+j 2π f1 T)` carrier — the matched
  filter / adjoint of the forward model.
- **Storage** (`results.save_run`): per-run folder with chunked `results.h5`
  (recon volume, fm data, model cloud), `meta.json`, and a dB max-projection
  preview. `load_run(..., load_img=False)` lazily reads single slices.

## Quick start

### On the GPU server (`bose`)
```bash
cd ~/Documents/code/model_gen && git pull && cd python
conda activate ra                        # Python 3.10 env
# deps (once): cupy-cuda12x h5py matplotlib pillow numpy pytest  (pip)

python -m pytest -q forward_model_ref.py raforward.py recon.py   # validate
python bench.py                          # dual-precision timing
python run_model.py --n-points 4000000 --run-id body_4M          # full run
```

### Real data
```bash
python run_model.py --xyz /mnt/shared_drive/Jonah/synthetic_data/body.xyz --run-id real_pose
```

### Output location (`RA_OUTPUT_ROOT`)
Default = NAS mount on `bose`. Set per machine:
- `bose`:  `/mnt/shared_drive/Jonah/synthetic_data` (default)
- Mac:   `export RA_OUTPUT_ROOT=/Volumes/Shared_Drive/Jonah/synthetic_data`

Browse results on the Mac: open `<run>/preview/max_projections.png`, or build a
campaign index with `results.contact_sheet(...)`.

## Infrastructure

- **GPU server `bose`** — RTX A4000 (16 GB, CC 8.6), RHEL, `10.12.7.36`. SSH key
  auth + agent forwarding for git. Code at `~/Documents/code/model_gen`,
  conda env `ra` (`cupy-cuda12x` + h5py/matplotlib/pillow via pip).
- **NAS** — `//10.12.0.209/Shared_Drive` (CIFS, `vers=3.0,sec=ntlmssp`,
  permanent via `/etc/fstab` + credentials file). Results under
  `Jonah/synthetic_data/`. Mounted on Mac as `/Volumes/Shared_Drive`, on `bose`
  as `/mnt/shared_drive`.

## Precision: fp32 vs fp64

- **fp32 (default)** — ~1e-3 relative error, which is **the current production
  accuracy** (the MATLAB mex is all-float32). The error is the float32
  representation of multi-metre distances (~2e-7 m → ~1e-3 rad phase), *not* a
  position error — it behaves like a ~−54 dB image floor. ~30× faster than fp64
  on the A4000 (which runs fp64 at 1/32 rate).
- **fp64** — ~1e-6, the optional higher-accuracy path; slower on consumer GPUs.

The reconstruction's localization/resolution are unaffected by precision (set by
aperture + bandwidth); fp32 only adds the noise floor.

## Not done / roadmap

1. **Real input data** — no body cloud is in the repo (correct: data isn't
   committed). Need a real `body.xyz` (from `model_exports`) or a prior run's
   `model.mat` reachable from `bose`. A `load_model_mat()` helper (extract `TARG`
   from an existing run) is a quick add for an apples-to-apples check vs the
   MATLAB recon.
2. **Model-generation flow** — the real pose/asset pipeline (Mixamo export →
   `place_assets` → HPR cull, `simulator_mixamo_flow.m`) is **not ported**.
   `synthetic_body` is only a test fixture.
3. **NUFFT acceleration** — discussed (angular-spectrum, type-1/2); not needed
   yet since the brute-force kernel is already fast enough at current scale.
4. **Repo hygiene** — already-tracked MATLAB/VS cruft (`.vs/`, `.mexw64`, `.pdb`)
   still needs a `git rm --cached` cleanup; `.gitignore` already blocks new junk.

## File map

```
python/
  raforward.py          forward model (Scene, forward, CuPy kernel + NumPy mirror)
  recon.py              backprojection recon (ReconGrid, backproject)
  results.py            save_run / load_run / contact_sheet, RA_OUTPUT_ROOT
  forward_model_ref.py  float64 reference + float32 accuracy study
  run_model.py          end-to-end driver (CLI)
  bodies.py             load_xyz (real) + synthetic_body (fixture)
  bench.py              dual-precision GPU benchmark
  requirements.txt
  README.md             this file
```
