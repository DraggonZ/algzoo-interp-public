"""Do weak activations matter (Gao)? Use the account's predicted switches only at steps where the true pre-activation
is weak (|pre| < d) or only where it is strong (|pre| >= d); everywhere else the switch is the true one."""
import numpy as np
from sim_switches import Xs, Ls, Ps, on_pred, run_masked
from simlib import GROUPS, score

print(f"{'group':12s} {'d':>4s} {'weak: wrong':>11s} {'acc':>6s} {'strong: wrong':>13s} {'acc':>6s}")
for name in ("cancellers", "memory"):
    ns = GROUPS[name]
    for d in (0.5, 1.0, 2.0):
        res = []
        for weak in (True, False):
            def mask_of(t, pre, weak=weak):
                m = pre > 0; sel = (np.abs(Ps[:, t, ns]) < d) == weak
                m[:, ns] = np.where(sel, on_pred[:, t, ns], m[:, ns]); return m
            sel_all = (np.abs(Ps[:, :, ns]) < d) == weak
            res += [((on_pred[:, :, ns] != (Ps[:, :, ns] > 0)) & sel_all).mean(), score(Xs, Ls, run_masked(Xs, mask_of))["acc"]]
        print(f"{name:12s} {d:4.1f} {res[0]:11.3f} {res[1]:6.3f} {res[2]:13.3f} {res[3]:6.3f}")
