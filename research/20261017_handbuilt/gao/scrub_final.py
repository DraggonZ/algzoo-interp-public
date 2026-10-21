"""Refined claims, scored on fresh lists (seed 303; the refinements were chosen on seed 202). Each group alone, then all
at once (every group's value at every step taken from its own resampled list). Two versions of the account:
  role claims  - each group depends on what its role names: store (m1 value and place, m2), cancellers (m1 value and
                 place, m2 value and place, previous record, the number after the current largest), keep gate (m1
                 place), late gate (base), memory (kept numbers, current and previous input).
  memory claim - every group depends only on the kept numbers at their places plus the current and previous input."""
import scrub
from simlib import GROUPS, lists, run, score
X = lists(303, 100_000); L = run(X)[0]
ROLE = {"store+detector": ("m1pos", "m2"), "cancellers": ("prevrec", "after"), "keep gate": ("m1pos",), "late gate": (),
        "memory": ("cur", "prev")}
MEM = {g: ("asmemory", "cur", "prev") for g in GROUPS}
print(f"trained accuracy on these lists {(L.argmax(1) == X.argsort(1)[:, -2]).mean():.3f}")
for label, opts in (("role claims", ROLE), ("memory claim", MEM)):
    for g in list(GROUPS) + ["all"]:
        gs = list(GROUPS) if g == "all" else [g]
        s = score(X, L, run(X, scrub.scrubbed_values(X, gs, opts=opts))[0])
        print(f"{label:13s} {g:15s}: KL {s['kl']:.3f} acc {s['acc']:.3f} agree {s['agree']:.3f}", flush=True)
