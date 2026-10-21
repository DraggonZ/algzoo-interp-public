"""How much does the keep gate's exact threshold matter? Neuron 5 keeps its own value (from the weights, ReLU) but is
forced off one step after any non-record below c x the second-largest so far (it can only forget more, never keep more).
c = 0 is the network itself; 'records only' forgets every non-record. 100,000 lists (seed 8)."""
import numpy as np
from gatelib import lists, run, task_vars, acc_by_kind

X = lists(8, 100_000); V = task_vars(X); L0, P0, H0 = run(X)
rec = np.zeros((len(X), 10), bool); rec[:, 1:] = V["new"][:, :9].astype(bool)
m2 = np.zeros((len(X), 10)); m2[:, 1:] = V["m2"][:, :9]
xs = np.zeros((len(X), 10)); xs[:, 1:] = X[:, :9]
KEYS = ["all", "S before L (revealed)", "S after L (kept)", "S at 8", "S at 9", "S right after L"]
base = acc_by_kind(X, L0, KEYS)
print(" " * 34 + "  ".join(f"{k[:6]:>6s}" for k in KEYS))
print(f"{'intact':34s}" + "  ".join(f"{100 * base[k][0]:6.1f}" for k in KEYS))
for c in (0.8, 0.85, 0.9, 0.95, 1.0, "records only"):
    if c == "records only":
        mk = rec.copy()
    else:
        mk = rec | (xs >= c * m2)
    mk[:, 0] = True
    changed = ((H0[:, :, 5] > 0) & ~mk).mean()

    def hook(t, hp, pre, h, mk=mk):
        h[:, 5] = np.maximum(pre[:, 5], 0) * mk[:, t]; return h
    r = acc_by_kind(X, run(X, hook=hook)[0], KEYS)
    print(f"{f'forced off below {c} x second' if c != 'records only' else 'forced off after every non-record':34s}" +
          "  ".join(f"{100 * r[k][0]:6.1f}" for k in KEYS) + f"   (switches off {100 * changed:.2f}% of steps)")
