"""Do the cancellers depend on the previous record (the largest before the current one arrived)? Claim: cancellers'
base claim + m2's position + the previous record's value and position. On the trained network and on v18."""
import sys
import numpy as np
import scrub
from algzoo.handbuilt_v18 import build_v18
from simlib import lists, run, score
X = lists(202, 100_000)
if sys.argv[1] == "v18":
    P = tuple(np.asarray(p, float) for p in build_v18().params); P = (P[0], P[1], P[2] * 447.7); scrub.PARAMS = P
    scrub.GROUPS["cancellers"] = [4, 5, 6]
else:
    P = None
L = run(X, params=P)[0]
for opts in (("m2pos",), ("prevrec",)):
    s = score(X, L, run(X, scrub.scrubbed_values(X, ["cancellers"], opts={"cancellers": opts}), params=P)[0])
    print(f"{sys.argv[1]:8s} cancellers + {'+'.join(opts):8s}: KL {s['kl']:.3f} acc {s['acc']:.3f} agree {s['agree']:.3f}", flush=True)
