#!/usr/bin/env bash
# Overnight chain: (#2) definitive potential + paths + blind-spot + motion, then
# (#1) gun detectability. Each step is guarded so a failure doesn't abort the rest.
set -u
cd /home/jonah/Documents/code/model_gen/python
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ra

POSE="/mnt/shared_drive/Jonah/synthetic_data/26JUN19 2/pose_70"   # note: path has a space
D=4; R=90; S=0.5           # definitive potential: decimate 4, +-90 deg, 0.5 deg step

cov() {  # $1 = path angle; POSE quoted so the space in the path survives
  python -u experiments/coverage_vs_range.py --pose "$POSE" --mode cross \
    --decimate $D --ref-range $R --ref-step $S --path-deg "$1"
}

echo "###### OVERNIGHT START  $(date)"
echo "###### STEP 1/6: definitive potential + cross 30 (dec$D +-$R $S)"
cov 30 || echo "STEP1 FAILED"
echo "###### STEP 2/6: cross 20 (reuse cached potential)"
cov 20 || echo "STEP2 FAILED"
echo "###### STEP 3/6: cross 45"
cov 45 || echo "STEP3 FAILED"
echo "###### STEP 4/6: blind-spot map"
python -u experiments/blindspot_map.py --pose "$POSE" --decimate $D --ref-range $R --ref-step $S || echo "STEP4 FAILED"
echo "###### STEP 5/6: motion multi-look 30 (dec$D)"
python -u experiments/motion_multilook.py --pose "$POSE" --path-deg 30 --n-frames 101 --decimate $D --ref-range $R --ref-step $S || echo "STEP5 FAILED"
echo "###### STEP 6/6: gun detectability (multi-look, dec 8)"
python -u experiments/threat_detectability.py --pose "$POSE" --asset gun --decimate 8 --frames 17 --n-locations 30 || echo "STEP6 FAILED"
echo "###### OVERNIGHT DONE  $(date)"
