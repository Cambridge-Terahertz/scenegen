# Running model_gen (Python/CuPy port) on `tesla`

Handoff notes for someone who has never run this. `setup_tesla.sh` (same folder)
does the whole bootstrap; this explains what it does, what it *can't* do for you,
and how to drive the simulator once it's up.

`model_gen` here means the **Python/CuPy port on branch `python-port`**, under
`python/` — not the MATLAB flow and not Scott's C++ `model_gen_test_dist`.

---

## 0. What you need before you start

| | |
|---|---|
| SSH access to tesla | `10.12.0.141`, user = your own account. RTX 5090, 32 GB VRAM, 96 cores, 881 GB RAM. |
| GitHub access to the repo | `Cambridge-Terahertz/model_gen`, private. Easiest is **agent forwarding** — put `ForwardAgent yes` in your `~/.ssh/config` block for tesla and connect with your GitHub key loaded (`ssh-add -l` should list it). Then git on tesla borrows your laptop's key; nothing is copied to the box. |
| Scene data | **Not in the repo** (correctly — data isn't committed) and **tesla has no NAS mount**. See §4. |

## 1. Run the bootstrap

```bash
scp python/setup_tesla.sh tesla:~/
ssh -A tesla 'bash ~/setup_tesla.sh'
```

Takes a few minutes, mostly the miniconda + CuPy downloads. It's idempotent —
re-run it any time to pull the latest code and re-verify; steps already done are
skipped.

### What it does, and why each step exists

1. **GPU check + CUDA detection.** Reads the CUDA major version from `nvcc` (or
   falls back to `nvidia-smi`) and picks the matching CuPy wheel — `cupy-cuda13x`
   for CUDA 13, `cupy-cuda12x` for 12. Getting this wrong is the single most
   common failure; that's why the script derives it instead of hardcoding.
2. **Miniconda** into `~/miniconda3` if absent, then `conda tos accept` for the
   `main` and `r` channels. A fresh miniconda **refuses to solve any environment**
   until that ToS is accepted, with an error that doesn't say so clearly.
3. **Conda env `ra`** (Python 3.11) + deps via pip: numpy, scipy, matplotlib,
   h5py, pillow, pytest, trimesh, networkx, and CuPy.
   ⚠️ CuPy is installed as **`cupy-cuda13x[ctk]`** — the `[ctk]` extra is
   **required**. Without the bundled toolkit headers, CuPy's runtime NVRTC
   compile of the forward kernel fails with *"Failed to find CUDA headers"*.
   The script then does a real `cupy` import + tiny GPU op so you find out here
   rather than 90 s into a run.
4. **Clone/pull** the repo at `~/Documents/code/model_gen`, branch `python-port`.
5. **`RA_OUTPUT_ROOT`** — the important tesla-specific bit. `results.py` defaults
   to `/mnt/shared_drive/Jonah/synthetic_data`, which is the **bose** NAS mount.
   **tesla has no such mount**, so with the default every run computes fine and
   then dies at save time. The script points it at `~/synthetic_data` and writes
   the export into the env's `activate.d`, so it's set for every future
   `conda activate ra` — you don't have to remember it.
6. **Smoke test** — `pytest` on `raforward.py` + `recon.py`, then a 200k-point
   synthetic end-to-end run. If that prints `saved -> ...`, you're done.

## 2. Every session after that

```bash
ssh -A tesla
conda activate ra
cd ~/Documents/code/model_gen/python
```

## 3. Running the simulator

`run_model.py` is the end-to-end driver: cloud → forward model → backprojection
recon → saved run folder.

```bash
# synthetic body (no data needed) — good for checking the box still works
python run_model.py --n-points 4000000 --run-id smoke_4M

# a real scene folder (body/ + assets/)
python run_model.py --scene ~/synthetic_data/26JUN19/pose_70 \
                    --decimate 4 --assets gun --run-id pose70_gun

# a bare point cloud
python run_model.py --xyz ~/data/body.xyz --run-id real_pose
```

Useful flags: `--decimate N` (body stride — the main runtime knob),
`--precision fp32|fp64` (fp32 default and is what production accuracy already is;
fp64 is ~30× slower on consumer GPUs), `--voxel` (recon voxel size, default 1 cm),
`--nrx`/`--nf` (array size 130×130 = 16 900, 128 frequencies).

Each run writes `<RA_OUTPUT_ROOT>/<run-id>/` containing `results.h5` (recon
volume + fm data + cloud, chunked), `meta.json` (params + timings), and
`preview/max_projections.png`. Long runs: `nohup python -u run_model.py … &` or
tmux — an SSH drop otherwise kills the job.

**Rough scale, for expectation-setting:** on the A4000 a 4M-scatterer run is
~74 s (71.6 s forward + 2.8 s recon). The 5090 is substantially faster; forward
time scales roughly linearly with scatterer count, recon is fixed by grid size.

## 4. Getting data onto the box, and results off it

tesla has **no `/mnt/shared_drive`** — this is the main day-to-day difference
from bose. Everything moves by `scp`:

```bash
# scene in  (from the Mac, where the NAS is mounted at /Volumes/Shared_Drive)
scp -r "/Volumes/Shared_Drive/Jonah/synthetic_data/26JUN19/pose_70" tesla:~/synthetic_data/

# results out
scp -r tesla:~/synthetic_data/pose70_gun "/Volumes/Shared_Drive/Jonah/synthetic_data/"
```

Then browse `preview/max_projections.png` locally, or use
`results.load_run(..., load_img=False)` to read single slices without pulling the
whole volume.

## 5. Gotchas, in the order you'll hit them

- **`Failed to find CUDA headers`** → CuPy installed without `[ctk]`.
  `pip install "cupy-cuda13x[ctk]"`.
- **Run completes, then a save error about `/mnt/shared_drive`** → `RA_OUTPUT_ROOT`
  isn't set. Re-run the script, or `export RA_OUTPUT_ROOT=$HOME/synthetic_data`.
- **`git clone` asks for a password** → agent forwarding isn't working. Connect
  with `ssh -A tesla` and check `ssh-add -l` on your laptop first.
- **Conda hangs or errors on the first `create`** → channel ToS not accepted.
- **CuPy imports but kernels fail to compile** → the GPU is Blackwell (`sm_120`),
  which needs CUDA **12.8+**. CuPy compiles the forward kernel at runtime via
  NVRTC so you don't set an arch by hand — but an older toolkit will reject
  `sm_120`. (This is also why the C++ side differs per box: `build.sh` defaults to
  `CMAKE_CUDA_ARCHITECTURES=120`, correct on tesla, wrong on bose which needs 86.)
- **Writing your own kernels?** One thread per array element launches only ~17k
  threads and uses ~5 % of a 5090 — it wants ~350k in flight. Use a 2-D grid
  (element × frequency-tile). This took one experiment from >10 min to 49 s.

## 6. Known drift in the source — read before you trust a fresh clone

As of **2026-07-22**, a clean clone of `python-port` does **not** include
everything that exists elsewhere:

- **Forward-model optimizations are uncommitted on bose.** The `--use_fast_math`
  + `INV_TWO_PI` + optional phase-recursion work (~2.5× forward throughput,
  628 Geval/s with recursion on) lives only in bose's working tree. A fresh clone
  gets the slower-but-correct kernel. Verify with
  `grep -c PHASE_RECURSION python/raforward.py` — `0` means you have the old one.
- Local uncommitted edits to `bodies.py` / `run_model.py` exist on the Mac clone.
- `python/README.md` and this file are untracked; a clone won't have them unless
  they're committed or copied over.

If you need parity with what's been benchmarked, sync from the machine that has
the work rather than assuming `git pull` is enough.
