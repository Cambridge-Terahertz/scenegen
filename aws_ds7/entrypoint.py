"""AWS Batch array-task entrypoint: generate ONE body's dataset7 scans, upload to S3.

Batch sets AWS_BATCH_JOB_ARRAY_INDEX. One array index == one body, because the driver reuses the
posed SMPL-X mesh across all four views -- sharding per (body, view) instead would re-pay the
~96 s body prep four times. A body is 1,330 scans in `cross` mode, ~90 min on a g5.xlarge.

    body_index = DS7_BODY_OFFSET + AWS_BATCH_JOB_ARRAY_INDEX

Assets (SMPL-X, objects_v5, STLs) are license-restricted and NOT baked into the image; they come
from a private S3 tarball at runtime, extracted to /root, which is where the driver's ~-relative
defaults look. Extraction is skipped when the container is warm.

Env from the job definition:
    DS7_S3_BUCKET   (required)  destination bucket
    DS7_S3_PREFIX   default "dataset7"
    DS7_S3_ASSETS   (required)  s3:// uri of the asset tarball
    DS7_BODY_OFFSET default 0   added to the array index
    any other DS7_* knob (APER, ORIENTS, PLACEMENT, NPLACE, SEED, JITTER, ...) is passed straight
    through to the driver, so a pilot and the full run differ only by job-definition overrides.
"""
import os, subprocess, sys, glob

idx = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", os.environ.get("DS7_INDEX", "0")))
body = int(os.environ.get("DS7_BODY_OFFSET", "0")) + idx
bucket = os.environ["DS7_S3_BUCKET"]
prefix = os.environ.get("DS7_S3_PREFIX", "dataset7")
assets = os.environ["DS7_S3_ASSETS"]

# 1. license-restricted assets -> /root (idempotent; a warm container skips it)
if not os.path.exists("/root/smplx_models"):
    subprocess.check_call(["aws", "s3", "cp", assets, "/tmp/assets.tgz"])
    subprocess.check_call(["tar", "xzf", "/tmp/assets.tgz", "-C", "/root"])

# 2. run exactly this one body, all views
out = f"/tmp/ds7_{body:03d}"
os.makedirs(out, exist_ok=True)
env = dict(os.environ, DS7_OUT=out, DS7_BODY0=str(body), DS7_NBODIES="1")
print(f"[batch idx {idx}] body={body} -> {out}", flush=True)
subprocess.check_call(["python3", "/root/ds7/dataset7_driver.py"], env=env)

# 3. upload this body's scans. Fail loudly on an empty result rather than reporting success --
# a task that silently uploads nothing looks identical to a task that had no work to do.
made = glob.glob(f"{out}/*.npz")
if not made:
    sys.exit(f"driver produced no npz in {out}")
dst = f"s3://{bucket}/{prefix}/body_{body:03d}/"
subprocess.check_call(["aws", "s3", "cp", out, dst, "--recursive", "--only-show-errors"])
print(f"[batch idx {idx}] uploaded {len(made)} scans -> {dst}", flush=True)
