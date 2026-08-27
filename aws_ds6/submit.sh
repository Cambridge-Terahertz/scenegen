#!/usr/bin/env bash
# Register the dataset6 Batch job definition and submit a run. Run from the Mac (or anywhere with
# the jonah-cli creds). Costs nothing until a job is actually submitted.
#
#   bash submit.sh register              # (re)register the job definition only
#   bash submit.sh smoke                 # 1 body, 3 placements/view, on-demand  (~$1, ~5 min)
#   bash submit.sh pilot [N]             # N bodies (default 24), on-demand      (~$35)
#   bash submit.sh full  [N]             # N bodies (default 200)                (~$299 OD)
#   DS6_QUEUE=thz-queue bash submit.sh pilot     # ...on the SPOT queue instead
#
# One array index == one body. The driver reuses the posed mesh across all four views, so a
# finer shard would re-pay the ~96 s body prep per view.
set -euo pipefail

ACCT=980885145101
REGION=us-east-1
REPO=thz-dataset6
BUCKET=thz-walkthrough-${ACCT}
IMAGE=${ACCT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:latest
JOBDEF=thz-dataset6
QUEUE=${DS6_QUEUE:-thz-queue-od}          # on-demand by default: g5 spot is only ~35% off today
AWS=${AWS:-aws}
WHAT=${1:-register}
N=${2:-}

register() {
  "$AWS" batch register-job-definition \
    --job-definition-name "$JOBDEF" --type container --region "$REGION" \
    --container-properties "$(cat <<JSON
{
  "image": "$IMAGE",
  "vcpus": 4, "memory": 14000,
  "resourceRequirements": [{"type": "GPU", "value": "1"}],
  "environment": [
    {"name": "DS6_S3_BUCKET", "value": "$BUCKET"},
    {"name": "DS6_S3_PREFIX", "value": "dataset6"},
    {"name": "DS6_S3_ASSETS", "value": "s3://$BUCKET/assets/ds6_assets.tgz"},
    {"name": "DS6_ORIENTS",   "value": "front,back,left,right"},
    {"name": "DS6_APER",      "value": "rect"},
    {"name": "DS6_FREQ_WINDOW","value": "taylor"},
    {"name": "DS6_PLACEMENT", "value": "cross"},
    {"name": "DS6_JITTER",    "value": "1"},
    {"name": "DS6_SEED",      "value": "20260825"}
  ]
}
JSON
)" --query 'revision' --output text
}

submit() {                                  # submit <name> <n_bodies> [extra-env-json]
  local name=$1 n=$2 extra=${3:-}
  local arrayspec=()
  [ "$n" -gt 1 ] && arrayspec=(--array-properties "size=$n")
  "$AWS" batch submit-job --job-name "$name" --job-queue "$QUEUE" \
    --job-definition "$JOBDEF" --region "$REGION" "${arrayspec[@]}" \
    ${extra:+--container-overrides "$extra"} \
    --query '[jobName,jobId]' --output text
}

# ---------------------------------------------------------------- staleness guard ----
# The image BAKES the driver. A stale image runs old code while everything else has moved on --
# which is exactly how a frame migration would silently produce 266,000 wrong-frame scans. Compare
# the pushed image against the driver's mtime and refuse rather than warn.
check_fresh() {
  local drv="$(cd "$(dirname "$0")" && pwd)/../dataset6_driver.py"
  [ -f "$drv" ] || return 0
  local pushed drv_epoch img_epoch
  pushed=$("$AWS" ecr describe-images --repository-name "$REPO" --region "$REGION" \
           --query 'sort_by(imageDetails,&imagePushedAt)[-1].imagePushedAt' --output text 2>/dev/null) || return 0
  [ -z "$pushed" ] || [ "$pushed" = None ] && return 0
  img_epoch=$(date -j -f "%Y-%m-%dT%H:%M:%S" "${pushed%%.*}" +%s 2>/dev/null \
              || date -d "${pushed%%.*}" +%s 2>/dev/null) || return 0
  drv_epoch=$(stat -f %m "$drv" 2>/dev/null || stat -c %Y "$drv")
  if [ "$drv_epoch" -gt "$img_epoch" ]; then
    echo "REFUSING: dataset6_driver.py is newer than the pushed ECR image." >&2
    echo "  driver $(date -r "$drv_epoch" 2>/dev/null || date -d @"$drv_epoch")" >&2
    echo "  image  $pushed" >&2
    echo "  Rebuild first:  ssh tesla 'cd ~/scenegen/aws_ds6 && bash build.sh'" >&2
    echo "  Override with FORCE_STALE=1 only if you know the difference does not matter." >&2
    [ "${FORCE_STALE:-0}" = 1 ] || exit 2
  fi
}

case "$WHAT" in
  register) echo "registered revision $(register)" ;;
  smoke)
    check_fresh
    register >/dev/null
    # 3 placements per view and one body: proves image, assets, GPU, driver and the S3 round-trip
    # without paying for a real body. DS6_MAXPLACE truncates each view's placement list.
    submit ds6-smoke 1 '{"environment":[{"name":"DS6_MAXPLACE","value":"3"}]}' ;;
  pilot)
    check_fresh
    # FULL 17x36 cross: 1330 placements/body, 31,920 scans, ~$83 at the measured 9 s/scan on
    # g5.xlarge. User approved the cost 2026-08-25. Exhaustive object x site coverage, so the
    # sample-mode question ("does 100/view give the same gate answers?") can be settled by
    # subsampling this set offline instead of paying for a second run.
    register >/dev/null
    submit ds6-pilot "${N:-24}" '{"environment":[
        {"name":"DS6_PLACEMENT","value":"cross"},
        {"name":"DS6_S3_PREFIX","value":"dataset6_pilot"}]}' ;;
  pilot-sample)
    register >/dev/null
    submit ds6-pilot-s "${N:-24}" '{"environment":[
        {"name":"DS6_PLACEMENT","value":"sample"},
        {"name":"DS6_NPLACE","value":"100"},
        {"name":"DS6_S3_PREFIX","value":"dataset6_pilot_s"}]}' ;;
  full)
    check_fresh
    register >/dev/null
    submit ds6-full "${N:-200}" ;;
  *) echo "usage: $0 register|smoke|pilot|full [N]" >&2; exit 2 ;;
esac
