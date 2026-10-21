"""Positive control for the scrubbing test: v18, whose dependencies are known from its rules. Claims, by v18's roles:
  store+detector (S, D): x_t and the value of m1 only (the trained network's base claim) - exact for v18.
  keep (G): x_t, x_{t-1}, values of m1 and m2 before t-1 (base keep claim) - v18 writes (1 - c) x(t-1), or the previous
      largest for a record, minus 1e-4 m1, so exact.
  memory: kept numbers at their positions, with v18's rule "every positive number is kept" (threshold 0).
  clock: the trained cancellers' base claim (x_t, m1 value and position, m2 value) - NOT exact for v18: its wave has
      height d, the record's jump over the previous largest, which this claim does not keep. Expected to fail.
v18's scores are multiplied by 448 for KL (as in sim_v18.py). 100,000 lists of seed 202."""
import numpy as np
import scrub
from algzoo.handbuilt_v18 import build_v18
from simlib import lists, run, score
P = tuple(np.asarray(p, float) for p in build_v18().params); P = (P[0], P[1], P[2] * 447.7); scrub.PARAMS = P
scrub.GROUPS.clear(); scrub.GROUPS.update({"store+detector": [0, 1], "keep gate": [2], "memory": list(range(8, 16)),
                                           "cancellers": [4, 5, 6]})
X = lists(202, 100_000); L = run(X, params=P)[0]
print(f"v18 accuracy {(L.argmax(1) == X.argsort(1)[:, -2]).mean():.3f}")
for g, opts in (("store+detector", ()), ("keep gate", ()), ("memory", ("thr=0",)), ("cancellers", ()), ("cancellers", ("m2pos",))):
    s = score(X, L, run(X, scrub.scrubbed_values(X, [g], opts={g: opts}), params=P)[0])
    c = score(X, L, run(X, scrub.scrubbed_values(X, [g], control=True, opts={g: opts}), params=P)[0])
    print(f"{g:16s} {'+'.join(opts) or '(base)':8s} claim: KL {s['kl']:.3f} acc {s['acc']:.3f} agree {s['agree']:.3f}   control: acc {c['acc']:.3f}", flush=True)
