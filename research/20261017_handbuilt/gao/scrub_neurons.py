"""Which single neurons carry the features the claims missed (gao/scrub_refine.log)? Scrub one neuron at a time under
its group's base claim, then with the feature that fixed the group. 100,000 lists of seed 202."""
import scrub
from scrub import lists, run, score, scrubbed_values
X = lists(202, 100_000); L = run(X)[0]
FIX = {"store+detector": ("m1pos", "m2"), "cancellers": ("m2pos",), "keep gate": ("m1pos",)}
print(f"{'neuron':>6s} {'group':16s} {'base acc':>8s} {'fixed acc':>9s}  (fix adds)")
for g, fix in FIX.items():
    for n in scrub.GROUPS[g]:
        # claim_list dispatches on the group name, so pass the group's claim but substitute only neuron n
        base = {k: v for k, v in scrubbed_values(X, [g]).items() if k == n}
        fixd = {k: v for k, v in scrubbed_values(X, [g], opts={g: fix}).items() if k == n}
        b = score(X, L, run(X, base)[0])["acc"]; f = score(X, L, run(X, fixd)[0])["acc"]
        print(f"{n:6d} {g:16s} {b:8.3f} {f:9.3f}  ({'+'.join(fix)})", flush=True)
