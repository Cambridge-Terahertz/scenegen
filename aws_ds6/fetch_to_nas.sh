#!/usr/bin/env bash
# Pull a finished dataset6 run out of S3 onto the NAS, verify the count, then delete it from S3.
# Nothing is meant to live in S3 -- it is a staging buffer between the Batch tasks and the NAS.
#
# Run ON TESLA: it is wired to both the NAS (//10.12.0.209/Shared_Drive at /mnt/shared_drive) and
# to S3, so the bytes go straight across instead of transiting the Mac's SMB mount.
#
#   bash fetch_to_nas.sh                  # sync, verify, and PROMPT before deleting from S3
#   bash fetch_to_nas.sh --purge          # sync, verify, delete without prompting
#   DS6_PREFIX=dataset6_pilot bash fetch_to_nas.sh
set -euo pipefail

ACCT=980885145101
REGION=us-east-1
BUCKET=thz-walkthrough-${ACCT}
PREFIX=${DS6_PREFIX:-dataset6}
DEST=${DS6_NAS:-/mnt/shared_drive/Jonah/synthetic_data/dataset6_2026-08-25}
AWS=$(command -v aws || echo "${HOME}/miniconda3/envs/ra/bin/aws")
PURGE=${1:-}

[ -d "$(dirname "$DEST")" ] || { echo "FATAL: NAS not mounted at $(dirname "$DEST")" >&2; exit 1; }
mkdir -p "$DEST"

echo "== syncing s3://${BUCKET}/${PREFIX}/ -> ${DEST}"
"$AWS" s3 sync "s3://${BUCKET}/${PREFIX}/" "$DEST/" --region "$REGION" --only-show-errors

# Verify BEFORE deleting anything. Compare object counts, not just "sync exited 0" -- a partial
# sync that hit a permissions error on one prefix still exits 0 for the rest.
n_s3=$("$AWS" s3 ls "s3://${BUCKET}/${PREFIX}/" --recursive --region "$REGION" | grep -c '\.npz$' || true)
n_nas=$(find "$DEST" -name '*.npz' | wc -l | tr -d ' ')
echo "== s3: ${n_s3} npz   nas: ${n_nas} npz"
if [ "$n_s3" -eq 0 ]; then echo "nothing in S3 under ${PREFIX}/ -- already purged?"; exit 0; fi
if [ "$n_nas" -lt "$n_s3" ]; then
  echo "FATAL: NAS has fewer files than S3 -- NOT deleting. Re-run the sync." >&2; exit 1
fi
du -sh "$DEST"

if [ "$PURGE" != "--purge" ]; then
  read -r -p "delete s3://${BUCKET}/${PREFIX}/ ? [y/N] " ans
  [ "$ans" = y ] || { echo "kept in S3."; exit 0; }
fi
"$AWS" s3 rm "s3://${BUCKET}/${PREFIX}/" --recursive --region "$REGION" --only-show-errors
echo "== purged s3://${BUCKET}/${PREFIX}/  (assets tarball under assets/ is left in place)"
