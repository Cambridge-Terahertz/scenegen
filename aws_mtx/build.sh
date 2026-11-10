#!/usr/bin/env bash
# Build the co-array study image from tesla's ~/weekend_run (the exact code that made the weekend ML dataset) + the
# multi-Tx wrapper, push to ECR, and stage the license-restricted assets (SMPL-X, objects_v5, STLs) to private S3.
# Run ON tesla.  bash build.sh [all|image|assets]
set -euo pipefail
AWS=$(command -v aws || echo "${HOME}/miniconda3/envs/ra/bin/aws")
ACCT=980885145101; REGION=us-east-1; REPO=thz-mtx; BUCKET=thz-walkthrough-${ACCT}; ECR=${ACCT}.dkr.ecr.${REGION}.amazonaws.com
CTX=${HOME}/mtx_build; WR=${HOME}/weekend_run; WHAT=${1:-all}; HEREDIR=$(cd "$(dirname "$0")" && pwd)
rm -rf "$CTX"; mkdir -p "$CTX/weekend_run"; cp "$HEREDIR"/{Dockerfile,entrypoint.py} "$CTX"/
# code only: *.py and *.json from weekend_run (no meshes, no outputs)
( cd "$WR" && cp *.py *.json "$CTX/weekend_run/" )
for f in multitx_driver.py mtx_configs.py weekend_driver.py raforward.py recon.py; do [ -f "$CTX/weekend_run/$f" ] || { echo "FATAL missing $f"; exit 1; }; done
if [ "$WHAT" = all ] || [ "$WHAT" = image ]; then
  "$AWS" ecr describe-repositories --repository-names $REPO --region $REGION >/dev/null 2>&1 || "$AWS" ecr create-repository --repository-name $REPO --region $REGION >/dev/null
  "$AWS" ecr get-login-password --region $REGION | podman login --username AWS --password-stdin "$ECR"
  podman build -t "$REPO:latest" "$CTX"; podman tag "$REPO:latest" "$ECR/$REPO:latest"; podman push "$ECR/$REPO:latest"; echo "pushed $ECR/$REPO:latest"
fi
if [ "$WHAT" = all ] || [ "$WHAT" = assets ]; then
  [ -n "$(ls $WR/objects_v5/*.obj 2>/dev/null)" ] || { echo "FATAL: no meshes in $WR/objects_v5"; exit 1; }
  TGZ=/tmp/mtx_assets.tgz; tar czf "$TGZ" -C "$HOME" smplx_models weekend_run/objects_v5 weekend_run/glock17_ext.stl weekend_run/phone_hull.stl
  ls -lh "$TGZ"; "$AWS" s3 cp "$TGZ" "s3://${BUCKET}/assets/mtx_assets.tgz" --only-show-errors; echo "staged s3://${BUCKET}/assets/mtx_assets.tgz"
fi
