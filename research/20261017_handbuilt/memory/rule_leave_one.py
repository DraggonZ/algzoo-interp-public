"""Question: in the machine with all eight memory neurons switched by rules (sign_rules2.py), which neuron's rule costs
most in combination? Put one neuron (or one neuron at one step) back on its own sign, the other seven on rules;
closed loop. 100,000 lists (seed 11), rules fitted on seed 12."""
import numpy as np
from memlib import MEM, run, task, lists, score, kinds
from sign_rules2 import build_rules, apply_rules, machine
X = lists(11, 100_000); K = kinds(X); V = task(X); S0, P0, H0 = run(X); base = score(X, S0, S0, K)
R = build_rules(lists(12, 100_000)); off = apply_rules(R, X, V, H0); true_off = P0[:, :, MEM] <= 0
S = machine(X, H0, off, "closed"); a0 = score(X, S0, S)["all"][0]
print(f"all eight by rule: {100 * a0:.1f}; trained {100 * base['all'][0]:.1f}")
for j, n in enumerate(MEM):
    o = off.copy(); o[:, :, j] = true_off[:, :, j]
    r = score(X, S0, machine(X, H0, o, "closed"), K)
    per_step = []
    for t in range(10):
        o = off.copy(); o[:, t, j] = true_off[:, t, j]; per_step.append(100 * (score(X, S0, machine(X, H0, o, "closed"))["all"][0] - a0))
    print(f"  neuron {n:2d} back on its own sign: acc {100 * r['all'][0]:.1f} ({100 * (r['all'][0] - a0):+.1f} vs all-rule) | step 9 lists {100 * (r['new largest at step 9'][0] - base['new largest at step 9'][0]):+.1f}, step 8 {100 * (r['new largest at step 8'][0] - base['new largest at step 8'][0]):+.1f} vs trained | gain by step: " + " ".join(f"{x:+.1f}" for x in per_step))
