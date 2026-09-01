# dataset6 on AWS Batch

Runbook for generating `dataset6_2026-08-25` (ds6_m4v) in the cloud. Account **980885145101**,
region **us-east-1**. See `../ds6_scope.html` for why this dataset exists and what it fixes.

## Why the cloud at all

Both local GPU hosts are committed: tesla is running Silas's `full_frame_model` training, and
bose is saturated finishing dataset5's left/right views with 3.3 GB of root disk left. ds6 has
nowhere local to run. The G-instance quota was raised to **128 vCPU on-demand and 128 spot** on
2026-07-30, which is 32 concurrent `g5.xlarge` — enough to turn a 36-day single-GPU job into
one working day.

## Cost

Measured basis: **6.6 s/scan** on bose's A4000 with envelope recon (vs 24 s dense). AWS rates
below are scaled from that by fp32 throughput and are **estimates until the smoke job measures
them for real** — the main reason the pilot exists.

| | scans | g5 spot | g5 on-demand | g6e on-demand |
|---|---:|---:|---:|---:|
| smoke (1 body, 3 placements/view) | 12 | ~$1 | ~$1 | ~$1 |
| pilot (24 bodies) | 31,920 | $23 | $35 | $23 |
| full (200 bodies) | 266,000 | $194 | $299 | $190 |

Storage ~208 GB → $4.80/month in S3; ~$10 to egress the full set back to the Mac (first 100
GB/month free).

**g5 spot is only ~35% off today** ($0.653/hr median vs $1.006 on-demand), not the usual 70%, so
the default queue here is **on-demand** — it avoids interruption risk for almost no premium. If
the smoke job confirms L40S timing, `g6e.xlarge` on-demand matches g5 spot on price and finishes
in a third of the wall time; switch `instanceTypes` on the compute environment and re-run.

## Billing gate

The account has run **entirely on credits** since it opened ($122.31 over 14 months, almost all
of it a stopped MATLAB g5 deleted in July; August is $0.34). AWS exposes no credit-balance API —
read it at **Billing → Credits**. The July note put it at ~$52, which covers the smoke and the
pilot but **not** the full run. Before submitting `full`, confirm a valid card at
**Billing → Payment preferences**; this account has never actually been charged, so that path is
untested. Setting a Budget alert at the same time is cheap insurance: 32 concurrent g5s burn
~$32/hr if a job wedges in a retry loop.

## Layout

```
build.sh        assemble the context on tesla, build with podman, push to ECR, stage assets to S3
Dockerfile      CUDA 12.4 runtime + cupy-cuda12x + torch-cpu; code baked in, assets are not
entrypoint.py   one array index == one body: fetch assets, run all 4 views, upload to S3
submit.sh       register the job definition and submit smoke / pilot / full
```

Sharding is **one body per array task** because the driver reuses the posed SMPL-X mesh across
all four views — a per-(body, view) shard would re-pay the ~96 s body prep four times.

## Run it

```bash
# 1. on tesla — build the image and stage the assets (~15 min, CPU and network only)
ssh tesla 'cd ~/ds6_src/aws && bash build.sh'

# 2. from the Mac — prove the whole path end to end for about a dollar
bash submit.sh smoke

# 3. once smoke passes and the parity check below is clean
bash submit.sh pilot 24

# 4. only after the pilot clears the gate in ds6_scope.html §3
bash submit.sh full 200
```

Watch a run:

```bash
aws batch list-jobs --job-queue thz-queue-od --job-status RUNNING --region us-east-1
aws logs tail /aws/batch/job --follow --region us-east-1
aws s3 ls s3://thz-walkthrough-980885145101/dataset6/ --recursive --summarize | tail -3
```

## Parity check — do this before trusting 266,000 scans

Neither Batch queue retains job history from July, so the earlier walkthrough parity result
cannot be confirmed. Re-establish it: generate body 0 both in the cloud and on a local GPU with
identical env, then compare. `han_db` should agree bit-for-bit — the forward model and
backprojection are both fp32 and deterministic, and the placement rng is seeded from the
placement's identity rather than from wall time, so there is no legitimate source of drift.

```bash
python3 -c "
import numpy as np
a=np.load('cloud/ds6_m_h05_b10_front_0001_gun1_glock17_wb_front_mid.npz')
b=np.load('local/ds6_m_h05_b10_front_0001_gun1_glock17_wb_front_mid.npz')
print('han_db identical:', np.array_equal(a['han_db'], b['han_db']))
print('seat_xyz delta  :', np.abs(a['seat_xyz']-b['seat_xyz']).max())
"
```

A mismatch means the container's CuPy is dispatching a different kernel for the instance's SM
version — investigate before scaling, do not average it away.

## Gotchas already hit

- **cupy and torch need separate `pip install` lines.** A single `--index-url` sends cupy to the
  torch CPU index, where it does not exist. Cost a rebuild on the July image.
- **`awscli` is not on tesla's non-interactive PATH** — it lives in the `ra` conda env. `build.sh`
  resolves it explicitly.
- **The threat_dataset modules are spread across `~/probe`, `~/walkthrough` and
  `~/threat_dataset`** on tesla, with no single directory holding all of them. `build.sh` resolves
  each file by name against all three.
- **Asset paths are load-bearing.** `objects_v5/` must land at `/root/walkthrough/objects_v5`
  because `placement_config.py` resolves it relative to its own location, and
  `shapes_10x10.json` must sit next to the *driver* at `/root/ds6/`, not next to the modules.
- **Spot never provisioned in July** — a new account with no instance history is restricted. The
  quota is now approved at 128, but if spot still fails to place, use `thz-queue-od`.
