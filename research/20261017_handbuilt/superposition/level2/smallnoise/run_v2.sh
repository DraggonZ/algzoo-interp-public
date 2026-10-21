#!/bin/bash
export OMP_NUM_THREADS=3 PYTHONPATH=/data/output/src/algzoo/src
for s in 0 17; do /opt/venv/bin/python across_v2.py nets $s $((s+17 > 33 ? 33 : s+17)); done
for s in $(seq 0 20 199); do /opt/venv/bin/python across_v2.py noisy $s $((s+20)); done
for s in 0 20 40 60 80; do /opt/venv/bin/python across_v2.py shared $s $((s+20)); done
