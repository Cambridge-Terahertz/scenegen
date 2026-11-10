#!/bin/bash
cd ~/scenegen
for P in opt nominal onpanel; do
  PROBES=$P OUT=~/fourpole_out ~/miniconda3/envs/ra/bin/python -W ignore fourpole_fast.py
done
echo ALLDONE
