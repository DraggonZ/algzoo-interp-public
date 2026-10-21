"""The memory's claim with the current input kept ('cur'; the account forgets a number one step after it arrives), for
the memory and, as a union test, for the cancellers."""
import scrub
from simlib import lists, run, score
X = lists(202, 100_000); L = run(X)[0]
for g, opts in (("memory", ("cur",)), ("memory", ("cur", "prev")), ("cancellers", ("asmemory", "cur")), ("cancellers", ("asmemory", "cur", "prev"))):
    s = score(X, L, run(X, scrub.scrubbed_values(X, [g], opts={g: opts}))[0])
    print(f"{g:11s} + {'+'.join(opts):18s}: KL {s['kl']:.3f} acc {s['acc']:.3f} agree {s['agree']:.3f}", flush=True)
