"""When are memory neurons 0 and 15 off (pre-activation below zero)? Off-rate by step and by the largest's age, and
the main inputs to the pre-activation when off versus on."""
import numpy as np
from simlib import lists, run, task_vars, WH, WI
X = lists(202); _, P, H = run(X); V = task_vars(X)
np.set_printoptions(precision=2, suppress=True, linewidth=160)
for n in (0, 15):
    off = P[:, :, n] < 0
    print(f"neuron {n}: off {off.mean():.2f}; by step:", off.mean(0))
    age = V["age"]
    print("   by age of the largest (-1 = no positive yet, 0 = record now):", [f"{a}:{off[age == a].mean():.2f}" for a in range(-1, 10)])
    for cond, lab in ((V["newmax"] > 0, "new largest now"), (np.c_[np.zeros((len(X), 1), bool), V["newmax"][:, :-1] > 0], "new largest one step ago"), (V["x"] < 0, "input negative")):
        print(f"   off when {lab}: {off[cond].mean():.2f}; otherwise {off[~cond].mean():.2f}")
    # contributions to the pre-activation from each neuron (previous step) and input, mean when off vs on
    prev = np.concatenate([np.zeros((len(X), 1, 16)), H[:, :-1]], 1)
    contrib = prev * WH[n][None, None, :]
    print("   mean contribution when off (by source neuron):", contrib[off].mean(0))
    print("   mean contribution when on:                     ", contrib[~off].mean(0))
    print(f"   input term when off {(X * WI[n])[off].mean():.2f}, on {(X * WI[n])[~off].mean():.2f}")
