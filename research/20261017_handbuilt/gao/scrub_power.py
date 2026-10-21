"""How much does the memory claim (kept numbers + current and previous input) leave fixed? Share of the ten numbers
redrawn at the last step, and the matched random control (same count kept, random places) for memory and cancellers."""
import numpy as np
import scrub
from simlib import lists, run, score
X = lists(202, 100_000); L = run(X)[0]
X2 = scrub.claim_list(X, 9, "memory", ("cur", "prev"))
red = (X2 != X).sum(1); print(f"numbers redrawn at the last step: mean {red.mean():.2f} of 10; distribution {np.bincount(red, minlength=9)[:9] / len(X)}")
for g in ("memory", "cancellers"):
    o = ("cur", "prev") if g == "memory" else ("asmemory", "cur", "prev")
    c = score(X, L, run(X, scrub.scrubbed_values(X, [g], control=True, opts={g: o}))[0])
    print(f"{g}: matched random control acc {c['acc']:.3f} KL {c['kl']:.2f}", flush=True)
