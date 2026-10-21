"""What else do the trained cancellers depend on? Their claim + previous record + the number right after the current
largest ('after'), and the memory's claim applied to them (every kept number at its place: 'asmemory')."""
import scrub
from simlib import lists, run, score
X = lists(202, 100_000); L = run(X)[0]
for opts in (("prevrec", "after"), ("asmemory",), ("asmemory", "prev")):
    s = score(X, L, run(X, scrub.scrubbed_values(X, ["cancellers"], opts={"cancellers": opts}))[0])
    print(f"cancellers + {'+'.join(opts):14s}: KL {s['kl']:.3f} acc {s['acc']:.3f} agree {s['agree']:.3f}", flush=True)
