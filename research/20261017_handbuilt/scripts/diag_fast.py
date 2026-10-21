"""Why does the one-run estimate miss more than bisection? Compare the two terms for a few models."""
import itertools
import json
import sys

import numpy as np
import torch as th

sys.argv = ["x"]
from algzoo.core import example, zoo

exec(open("scripts/estimator_curve.py").read().split("rng = np.random.default_rng(1)")[0])  # load b_one_run and constants only
v2 = json.load(open("bias_formula_v2.json"))
GRID = [1.05, 1.15, 1.3, 1.6, 2.0, 3.0, 5.0]
for name, P in (("ARC", example()), ("zoo0", zoo(16, 10, 0)), ("zoo1", zoo(16, 10, 1))):
    a_fast = [np.mean(np.maximum(0, -b_one_run(P, PL, "23", g))) for g in GRID]
    b_fast = np.mean(np.maximum(0, -b_one_run(P, PL, "12", 1.6)))
    print(f"{name}: 23-term per grid (one run) {np.round(a_fast, 4)}; 12-term one run {b_fast:.4f} vs bisection {v2[name]['b']:.4f}; 23 weighted bisection {v2[name]['a']:.4f}")
