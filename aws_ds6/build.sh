#!/usr/bin/env bash
# Assemble the build context on tesla, build the ds6 image with podman, push to ECR, and stage
# the license-restricted assets to S3. Run ON tesla. Idempotent -- safe to re-run.
#
#   bash build.sh            # build + push image, and build/upload the asset tarball
#   bash build.sh image      # image only
#   bash build.sh assets     # assets only
set -euo pipefail

# awscli lives in the ra conda env on tesla and is not on a non-interactive PATH.
AWS=$(command -v aws || echo "${HOME}/miniconda3/envs/ra/bin/aws")
[ -x "$AWS" ] || { echo "FATAL: no aws cli found" >&2; exit 1; }

ACCT=980885145101
REGION=us-east-1
REPO=thz-dataset6
BUCKET=thz-walkthrough-${ACCT}
ECR=${ACCT}.dkr.ecr.${REGION}.amazonaws.com
CTX=${HOME}/ds6_build
SRC_DIRS=("${HOME}/probe" "${HOME}/walkthrough" "${HOME}/threat_dataset")
MG=${HOME}/Documents/code/model_gen/python
WHAT=${1:-all}

# The threat_dataset modules and assets are spread across ~/probe, ~/walkthrough and
# ~/threat_dataset on tesla, so resolve each by name against all three rather than assuming one.
pick() {
  for d in "${SRC_DIRS[@]}"; do [ -e "$d/$1" ] && { echo "$d/$1"; return 0; }; done
  echo "FATAL: cannot find $1 in ${SRC_DIRS[*]}" >&2; return 1
}

# ---------------------------------------------------------------- build context ----
HEREDIR=$(cd "$(dirname "$0")" && pwd)
rm -rf "$CTX"; mkdir -p "$CTX"/{threat_dataset,model_gen_python,ds6}
cp "$HEREDIR"/{Dockerfile,entrypoint.py} "$CTX"/
# frame.py MUST sit next to the driver: the driver does
#   sys.path.insert(0, dirname(abspath(__file__))); import frame
# so without this the container fails with ImportError before the first scan.
cp "$HEREDIR"/../dataset6_driver.py "$HEREDIR"/../frame.py "$CTX"/ds6/

# Python modules the driver imports. placement_config resolves confuser_site_map.json relative to
# ITSELF (its other candidate is a stale job tmp dir), so the map must land in the same directory.
for f in anatomy.py attach.py threats.py body.py mesh_sample.py placement_config.py \
         stance.py bodies_smplx.py confuser_site_map.json; do
  cp "$(pick "$f")" "$CTX/threat_dataset/"
done
# shapes_10x10.json must sit next to the DRIVER (its default is "{HERE}/shapes_10x10.json"),
# which is a different container directory from the modules.
cp "$(pick shapes_10x10.json)" "$CTX/ds6/"
cp "$MG"/*.py "$CTX/model_gen_python/"

# ---------------------------------------------------------------- image ----
if [ "$WHAT" = all ] || [ "$WHAT" = image ]; then
  "$AWS" ecr describe-repositories --repository-names $REPO --region $REGION >/dev/null 2>&1 \
    || "$AWS" ecr create-repository --repository-name $REPO --region $REGION >/dev/null
  "$AWS" ecr get-login-password --region $REGION \
    | podman login --username AWS --password-stdin "$ECR"
  podman build -t "$REPO:latest" "$CTX"
  podman tag "$REPO:latest" "$ECR/$REPO:latest"
  podman push "$ECR/$REPO:latest"
  echo "pushed $ECR/$REPO:latest"
fi

# ---------------------------------------------------------------- assets ----
# SMPL-X + the objects_v5 meshes + the two STLs. NOT amass -- ds6 uses the fixed screening_pose,
# so the ~300 GB motion corpus the walkthrough image needed is not required here.
if [ "$WHAT" = all ] || [ "$WHAT" = assets ]; then
  # Layout matters: it extracts to /root, and each consumer looks in a specific place.
  #   smplx_models/            -> SMPLX_MODELS default
  #   walkthrough/objects_v5/  -> BOTH placement_config (_HERE/objects_v5, _HERE=/root/walkthrough)
  #                               AND the driver (now searches ~/walkthrough/objects_v5)
  #   walkthrough/*.stl        -> driver's GLOCK_STL / PHONE search
  STAGE=$(mktemp -d); mkdir -p "$STAGE/walkthrough"
  cp -r "$(pick objects_v5)" "$STAGE/walkthrough/"
  cp "$(pick glock17_ext.stl)" "$(pick phone_hull.stl)" "$STAGE/walkthrough/"
  TGZ=/tmp/ds6_assets.tgz
  tar czf "$TGZ" -C "$HOME" smplx_models -C "$STAGE" walkthrough
  rm -rf "$STAGE"
  ls -lh "$TGZ"
  "$AWS" s3 cp "$TGZ" "s3://${BUCKET}/assets/ds6_assets.tgz"
  echo "staged s3://${BUCKET}/assets/ds6_assets.tgz"
fi
