#!/usr/bin/env bash
# Register the co-array job definition and submit. Run from the Mac.
#   bash submit.sh smoke     # config C, body 18, 2 placements (~$0.5)
#   bash submit.sh run       # A,B,C,E x 4 held-out bodies (A in 4 shards/body, others 2) on-demand
set -euo pipefail
ACCT=980885145101; REGION=us-east-1; BUCKET=thz-walkthrough-${ACCT}; IMAGE=${ACCT}.dkr.ecr.${REGION}.amazonaws.com/thz-mtx:latest
JOBDEF=thz-mtx; QUEUE=${MTX_QUEUE:-thz-queue-od}; WHAT=${1:-register}
DS7='{"name":"DS7_ORIENTS","value":"front"},{"name":"DS7_OBJECTS","value":"gun1_glock17,gun2_glock19,gun3_hellcat,knife_switchblade,knife_folding,boxcutter,phone_modern,wallet,keys,badge"},{"name":"DS7_NOBASE","value":"1"},{"name":"DS7_ARMS","value":"up"},{"name":"DS7_APER","value":"rect"},{"name":"DS7_NFREQ","value":"64"},{"name":"DS7_ROI","value":"0.25"},{"name":"DS7_SIG_EVERY","value":"1"},{"name":"DS7_D","value":"0.536"},{"name":"DS7_PITCH","value":"0.0025"},{"name":"DS7_GSTEP","value":"0.0019"}'
aws batch register-job-definition --job-definition-name $JOBDEF --type container --region $REGION --retry-strategy attempts=2 \
  --container-properties "{\"image\":\"$IMAGE\",\"vcpus\":4,\"memory\":15000,\"resourceRequirements\":[{\"type\":\"GPU\",\"value\":\"1\"}],
   \"environment\":[{\"name\":\"MTX_S3_BUCKET\",\"value\":\"$BUCKET\"},{\"name\":\"MTX_S3_ASSETS\",\"value\":\"s3://$BUCKET/assets/mtx_assets.tgz\"},$DS7]}" --query revision --output text >/dev/null
sub() { local name=$1 tasks=$2 extra=${3:-}; local n=$(echo "$tasks" | tr ',' '\n' | wc -l | tr -d ' ')
  local arr=(); [ "$n" -gt 1 ] && arr=(--array-properties "size=$n")
  aws batch submit-job --job-name "$name" --job-queue $QUEUE --job-definition $JOBDEF --region $REGION "${arr[@]}" \
    --container-overrides "{\"environment\":[{\"name\":\"MTX_TASKS\",\"value\":\"$tasks\"}${extra}]}" --query '[jobName,jobId]' --output text; }
case "$WHAT" in
  smoke) sub mtx-smoke "C:18:0:1" ',{"name":"DS7_MAXPLACE","value":"2"},{"name":"MTX_S3_PREFIX","value":"mtx_smoke"}' ;;
  run) T=""; for c in A B C E; do ns=2; [ $c = A ] && ns=4; for b in 18 72 127 181; do for s in $(seq 0 $((ns-1))); do T="$T,$c:$b:$s:$ns"; done; done; done
       sub mtx-run "${T#,}" ',{"name":"MTX_S3_PREFIX","value":"mtx"}' ;;
  more) T=""; for c in RA536 RA268; do for b in 2 4 13 22 31 40 49 58 67 76 85 94 102 104 113 122 131 140 149 158 167 176 185 194; do for s in 0 1; do T="$T,$c:$b:$s:2"; done; done; done
       sub lc-more "${T#,}" ',{"name":"MTX_S3_PREFIX","value":"lc_more"}' ;;
  register) echo registered ;;
esac
