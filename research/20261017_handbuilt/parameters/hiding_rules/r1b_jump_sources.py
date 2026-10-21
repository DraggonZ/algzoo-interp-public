"""r1b diagnostic: which source neurons carry the previous-jump signal into each hiding neuron? For hiding neuron j,
split ARC's recurrent input into per-source contributions WH[j,k] h_k(t-1) for the 12 non-hiding sources, regress each
on the rule's regressors (features + blind hiding feedback, as in constrained_lean), and print the coefficient on the
previous jump. The per-source coefficients sum to the jump coefficient a one-step fit gives (non-hiding part)."""
import sys
sys.path.insert(0, "rules")
import numpy as np
from common import HID, WI, WH, account_features, sd_rule, task_vars, lag
import simlib

X = simlib.lists(101, 50_000); _, P, H = simlib.run(X)
V = task_vars(X); sd = sd_rule(X)
Hp = np.concatenate([np.zeros((len(X), 1, 16)), H[:, :-1]], 1)
for j in HID:
    A = account_features(X, j)
    F = np.stack([A[..., 1], A[..., 2], A[..., 3], A[..., 4], sd[..., 0], sd[..., 1], lag(V["jump"]), lag(V["jump"], 2),
                  lag(V["newmax"] * V["x"], 2)] + [Hp[..., k] - lag(X) * WI[k] for k in HID], -1).reshape(-1, 13)
    on = (P[:, :, j] > 0).reshape(-1)
    out = []
    for k in range(16):
        if k in HID:
            continue
        y = (WH[j, k] * Hp[..., k]).reshape(-1)
        c = np.linalg.lstsq(F[on], y[on], rcond=None)[0]
        out.append((k, c[6]))
    tot = sum(c for _, c in out)
    print(f"neuron {j}: prev-jump coefficient from non-hiding sources, on steps: total {tot:+.2f}; " +
          ", ".join(f"{k}:{c:+.2f}" for k, c in sorted(out, key=lambda z: -abs(z[1]))[:6]))
print("source activity vs previous jump (slope over all steps; fraction of steps on):")
J = lag(V["jump"]).reshape(-1)
for k in range(16):
    h = Hp[..., k].reshape(-1)
    print(f"  {k}: slope {np.polyfit(J, h, 1)[0]:+.2f}, on {np.mean(h > 0):.2f}, on after a record {np.mean(h[J > 0] > 0):.2f}")
