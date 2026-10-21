"""Question: how is an overtaken record's wave kept off the positions after the overtaking, position by position?

Lists as bridge/tail_cancel2.py: a record at r (> 0.3, at least 0.05 above everything earlier), nothing above it until
q, a new largest at q at least 0.05 above it, nothing above that after q. Two nudges, per unit:
  'both' : x_r and x_q together (the overtaken record grows, the new record's jump is unchanged) - tail_cancel2's nudge;
  'r'    : x_r alone (the jump at q shrinks by the same amount).
Exact split (decomp.py) of the final scores into memory writes by source and step, plus the other neurons' readout.
Printed: total; memory writes made at steps up to q ('before': the record's wave as written while it was the
largest, carried on through the real on/off pattern), at q+1, q+2, and later; the other neurons' own readout; and the
'before' part split by writer. 1,500,000 lists, seed 3 (the lists of tail_cancel2)."""
import numpy as np
from memlib import lists
from decomp import decompose, SOURCES

X = np.random.default_rng(3).standard_normal((1_500_000, 10))
f = lambda v: " ".join(f"{x:6.1f}" for x in v)
for r, q in [(1, 4), (2, 5), (2, 7), (4, 7)]:
    pre = np.maximum(X[:, :r].max(1), 0) if r else np.zeros(len(X))
    ok = (X[:, r] > pre + 0.05) & (X[:, r] > 0.3) & (X[:, r + 1:q].max(1) < X[:, r]) & (X[:, q] > X[:, r] + 0.05)
    if q < 9: ok &= X[:, q + 1:].max(1) < X[:, q]
    Xs = X[ok][:200_000]
    for nudge in ("both", "r"):
        dX = np.zeros_like(Xs); dX[:, r] = 1
        if nudge == "both": dX[:, q] = 1
        R = decompose(Xs, dX); m = R["same"]; C = R["C"][m]
        print(f"=== record at {r}, overtaken at {q}, nudge {nudge}: {len(Xs)} lists, {m.mean():.2f} kept (pattern unchanged); check {R['check']:.1e}")
        print(f"   positions                  " + " ".join(f"{k:6d}" for k in range(10)))
        print(f"   total                      {f(R['dS'][m].mean(0))}")
        print(f"   memory, writes up to step q {f(C[:, :, :q + 1].sum((1, 2)).mean(0))}")
        for t in range(q + 1, 10):
            print(f"   memory, writes at step {t}   {f(C[:, :, t].sum(1).mean(0))}")
        print(f"   other neurons' readout     {f(R['direct'][m].mean(0))}")
        print("   memory writes up to step q, by writer:")
        for i, s in enumerate(SOURCES):
            v = C[:, i, :q + 1].sum(1).mean(0)
            if np.abs(v).max() > 2: print(f"     {s:12s}             {f(v)}")
        for t in range(q + 1, min(q + 3, 10)):
            print(f"   memory writes at step {t} (q+{t - q}), by writer:")
            for i, s in enumerate(SOURCES):
                v = C[:, i, t].mean(0)
                if np.abs(v).max() > 2: print(f"     {s:12s}             {f(v)}")
