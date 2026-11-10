"""AWS Batch array task for the co-array hardware study: one (config, body, placement-shard) per array index.
MTX_TASKS = "cfg:body:shard:nshard,..." ; index picks one. Runs ~/weekend_run/multitx_driver.py (the unchanged
weekend study driver + multi-Tx geometry) and uploads the npz to s3://$BUCKET/$PREFIX/<cfg>/b<body>/."""
import os, subprocess, sys, glob
idx = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", os.environ.get("MTX_INDEX", "0")))
cfg, body, sh, nsh = os.environ["MTX_TASKS"].split(",")[idx].split(":")
bucket, prefix, assets = os.environ["MTX_S3_BUCKET"], os.environ.get("MTX_S3_PREFIX", "mtx"), os.environ["MTX_S3_ASSETS"]
if not os.path.exists("/root/smplx_models"):
    subprocess.check_call(["aws", "s3", "cp", assets, "/tmp/assets.tgz", "--only-show-errors"])
    subprocess.check_call(["tar", "xzf", "/tmp/assets.tgz", "-C", "/root"])
out = f"/tmp/mtx_{cfg}_{body}_{sh}"; os.makedirs(out, exist_ok=True)
env = dict(os.environ, DS7_OUT=out, DS7_BODY0=body, DS7_NBODIES="1", MTX_CFG=cfg, MTX_SHARD=sh, MTX_NSHARD=nsh)
print(f"[idx {idx}] cfg {cfg} body {body} shard {sh}/{nsh}", flush=True)
if cfg.startswith("RA"):     # plain study system (1 probe + Rx RA): RA536 / RA268 -> unchanged weekend_driver via the wrapper with 1 probe
    D = {"RA536": ("0.536", "0.0019"), "RA268": ("0.268", "0.0035"), "RA134": ("0.134", "0.0075")}[cfg]
    env.update(DS7_D=D[0], DS7_GSTEP=D[1], MTX_CFG="BASE" + cfg[2:])
subprocess.check_call(["python3", "/root/weekend_run/multitx_driver.py"], env=env, cwd="/root/weekend_run")
made = glob.glob(f"{out}/*.npz")
if not made: sys.exit(f"driver produced no npz in {out}")
dst = f"s3://{bucket}/{prefix}/{cfg}/b{body}/"
subprocess.check_call(["aws", "s3", "cp", out, dst, "--recursive", "--only-show-errors"])
print(f"[idx {idx}] uploaded {len(made)} scans -> {dst}", flush=True)
