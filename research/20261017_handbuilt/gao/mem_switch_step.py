"""Memory switches from the account's features, fitted separately for each step (the account's register is the same
at every step). Switch-only test as in sim_switches.py, for the memory and for all 16 (others as before)."""
import numpy as np
from simlib import lists, run, score
from sim_groups import account_features
from sim_switches import run_masked, on_pred as on_shared
Xf, Xs = lists(101), lists(202)
_, Pf, _ = run(Xf); Ls, Ps, _ = run(Xs)
MEM = [0, 3, 10, 11, 12, 13, 14, 15]; on_step = on_shared.copy()
for n in MEM:
    Ff, Fs = account_features(Xf, n), account_features(Xs, n)
    for t in range(10):
        co = np.linalg.lstsq(Ff[:, t], Pf[:, t, n], rcond=None)[0]; on_step[:, t, n] = Fs[:, t] @ co > 0
for label, on in (("shared fit", on_shared), ("per-step fit", on_step)):
    for name, ns in (("memory", MEM), ("all 16", list(range(16)))):
        def mask_of(t, pre, ns=ns, on=on):
            m = pre > 0; m[:, ns] = on[:, t, ns]; return m
        s = score(Xs, Ls, run_masked(Xs, mask_of))
        print(f"{label:12s} {name:7s} wrong switches {(on[:, :, ns] != (Ps[:, :, ns] > 0)).mean():.3f}  KL {s['kl']:.3f}  acc {s['acc']:.3f}")
