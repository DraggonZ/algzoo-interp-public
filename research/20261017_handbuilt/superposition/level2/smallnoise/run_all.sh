#!/bin/bash
# Runs across.py in chunks of about five minutes each (shared GPU, under 11 GB).
export OMP_NUM_THREADS=3 PYTHONPATH=/data/output/src/algzoo/src
for s in 0 10 20 30; do /opt/venv/bin/python across.py nets $s $((s+10 > 33 ? 33 : s+10)); done
for s in $(seq 0 25 199); do /opt/venv/bin/python across.py noisy $s $((s+25)); done
