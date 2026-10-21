"""Same test as g24_forget_more.py on new111's keep gate (neuron 10, the only other single-neuron keep gate found,
g22_across_rules.log): forced off one step after any non-record below c x the second-largest so far. 100,000 lists."""
import numpy as np
from gatelib import lists, run, task_vars, acc_by_kind, other_models

params = other_models()["new111"]; g = 10
X = lists(8, 100_000); V = task_vars(X); L0, P0, H0 = run(X, params=params)
rec = np.zeros((len(X), 10), bool); rec[:, 1:] = V["new"][:, :9].astype(bool)
m2 = np.zeros((len(X), 10)); m2[:, 1:] = V["m2"][:, :9]
xs = np.zeros((len(X), 10)); xs[:, 1:] = X[:, :9]
KEYS = ["all", "S before L (revealed)", "S after L (kept)", "S at 8", "S right after L"]
base = acc_by_kind(X, L0, KEYS)
print(f"{'new111 intact':34s}" + "  ".join(f"{k[:6]} {100 * base[k][0]:5.1f}" for k in KEYS))
for c in (0.85, 0.9, 0.95, 1.0):
    mk = rec | (xs >= c * m2); mk[:, 0] = True

    def hook(t, hp, pre, h, mk=mk):
        h[:, g] = np.maximum(pre[:, g], 0) * mk[:, t]; return h
    r = acc_by_kind(X, run(X, params=params, hook=hook)[0], KEYS)
    print(f"{f'forced off below {c} x second':34s}" + "  ".join(f"{k[:6]} {100 * r[k][0]:5.1f}" for k in KEYS))
