"""Question: what do neurons 3 and 15 add to the scores, and at which positions?
For each memory neuron: (1) its readout weights by position; (2) accuracy by position of the second-largest when the
neuron's final value is replaced by its mean over lists in the readout only (its influence on other neurons kept),
and (3) when it is held at its per-step mean during the whole run (mean ablation). Other memory neurons as
comparison. 300,000 lists, seed 31."""
import numpy as np
from memlib import MEM, WO, run, lists, kinds

X = lists(31, 300_000); y = X.argsort(1)[:, -2]; S0, P0, H0 = run(X)
acc_by_pos = lambda S: np.array([100 * (S.argmax(1) == y)[y == k].mean() for k in range(10)])
base = acc_by_pos(S0)
print("readout weights W_oh[position, neuron]:")
for n in MEM: print(f"  neuron {n:2d}: " + " ".join(f"{w:6.2f}" for w in WO[:, n]) + f"   | mean final value {H0[:, 9, n].mean():5.2f}, off at the end {100 * (H0[:, 9, n] == 0).mean():.0f}%")
print("\naccuracy by position of the second-largest (0-9), and overall; trained:")
print("  trained                       " + " ".join(f"{a:5.1f}" for a in base) + f" | all {100 * (S0.argmax(1) == y).mean():.1f}")
means = H0.mean(0)
for n in MEM:
    S1 = S0 + np.outer(means[9, n] - H0[:, 9, n], WO[:, n])
    def hook(t, pre, h, n=n):
        h = h.copy(); h[:, n] = means[t, n]; return h
    S2 = run(X, hook)[0]
    a1, a2 = acc_by_pos(S1), acc_by_pos(S2)
    print(f"  neuron {n:2d} readout at mean   " + " ".join(f"{a - b:+5.1f}" for a, b in zip(a1, base)) + f" | all {100 * (S1.argmax(1) == y).mean():.1f}")
    print(f"  neuron {n:2d} held at its mean  " + " ".join(f"{a - b:+5.1f}" for a, b in zip(a2, base)) + f" | all {100 * (S2.argmax(1) == y).mean():.1f}")
