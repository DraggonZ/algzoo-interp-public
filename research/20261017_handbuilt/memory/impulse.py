"""Question: what writes a record's wave and a kept number's vote into the memory, into which modes, and how is each
read out? Nudge one number (exact split by decomp.py; mean over lists whose on/off pattern the nudge leaves alone).
(a) a record at r that stays the largest to the end (hidden); (b) a kept number at s after the largest, which ends
as the second-largest (its vote). Per unit of the nudged number:
  1. the change of each other neuron's activity by step (the writers' time course);
  2. the memory's content in the five mode families (modes.log) by step: size of the mode's complex amplitude,
     with unit-length eigenvectors;
  3. final scores at positions 0-9 from the memory's writes by source (summed over steps), from the other neurons'
     own readout, and in total;
  4. the same final memory scores split by the step of the write, for the two largest writers.
1,000,000 lists, seed 21."""
import numpy as np
from memlib import MEM, OTH, A, lists
from decomp import decompose, SOURCES

np.set_printoptions(precision=1, suppress=True, linewidth=180)
lam, Vv = np.linalg.eig(A); Vv = Vv / np.linalg.norm(Vv, axis=0); Vinv = np.linalg.inv(Vv)
fam = {}
for j, l in enumerate(lam):
    fam.setdefault(round(abs(np.degrees(np.angle(l)))), []).append(j)
X = lists(21, 1_000_000)
pm = np.maximum(np.maximum.accumulate(np.c_[np.zeros(len(X)), X], 1)[:, :-1], 0)


def report(sel, pos, title):
    Xs = X[sel]; dX = np.zeros_like(Xs); dX[np.arange(len(Xs)), pos] = 1
    R = decompose(Xs, dX); ok = R["same"]
    print(f"\n=== {title}: {len(Xs)} lists, {ok.mean():.2f} with the on/off pattern unchanged (used); exact-split check {R['check']:.1e}")
    dO = R["dH"][ok][:, :, OTH]
    print("  1. change of each other neuron's activity by step (mean per unit):")
    for i, n in enumerate(OTH):
        print(f"     neuron {n:2d}:", " ".join(f"{x:6.1f}" for x in dO[:, :, i].mean(0)))
    z = np.einsum("jm,ntm->ntj", Vinv, R["dH"][ok][:, :, MEM])
    print("  2. memory content by mode family (size of complex amplitude, mean), by step:")
    for f, js in sorted(fam.items()):
        print(f"     {f:4d} deg:", " ".join(f"{x:6.1f}" for x in np.abs(z[:, :, js[0]]).mean(0)))
    C = R["C"][ok]; print("  3. final scores, positions 0-9 (mean per unit):")
    for i, s in enumerate(SOURCES):
        v = C[:, i].sum(1).mean(0)
        if np.abs(v).max() > 1: print(f"     memory writes by {s:12s}", " ".join(f"{x:6.1f}" for x in v))
    print(f"     memory, all writes        ", " ".join(f"{x:6.1f}" for x in C.sum((1, 2)).mean(0)))
    print(f"     other neurons' readout    ", " ".join(f"{x:6.1f}" for x in R["direct"][ok].mean(0)))
    print(f"     total                     ", " ".join(f"{x:6.1f}" for x in R["dS"][ok].mean(0)))
    big = np.argsort(-np.abs(C.sum(2).mean(0)).sum(1))[:3]
    for i in big:
        print(f"  4. memory scores from {SOURCES[i]} by step of the write (rows) at positions 0-9:")
        for t in range(10):
            v = C[:, i, t].mean(0)
            if np.abs(v).max() > 1: print(f"     step {t}:", " ".join(f"{x:6.1f}" for x in v))


for r in (2, 5):
    sel = (X[:, r] > pm[:, r] + 0.05) & (X[:, r] > 0.3) & (X[:, r + 1:].max(1) < X[:, r] - 0.05)
    report(sel, r, f"(a) record at {r} that stays the largest")
o = X.argsort(1)
for s in (5, 7):
    sel = (o[:, -2] == s) & (o[:, -1] < s) & (X[:, s] > 0.3)
    report(sel, s, f"(b) second-largest at {s}, after the largest")
