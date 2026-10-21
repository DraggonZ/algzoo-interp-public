"""Debugger loop on the scrubbing claims (gao/scrub.py): add one feature at a time to a claim and see what restores
the output. Base claims as in scrub.py; 100,000 lists of seed 202."""
from scrub import lists, run, score, scrubbed_values
X = lists(202, 100_000); L = run(X)[0]
VAR = {"store+detector": [(), ("m1pos",), ("m2",), ("m1pos", "m2"), ("m1pos", "m2", "prev")],
       "cancellers": [(), ("m2pos",), ("prev",), ("m2pos", "prev"), ("m2pos", "prev", "prev2")],
       "keep gate": [(), ("m1pos",), ("prev2",), ("m1pos", "prev2")],
       "memory": [(), ("prev",), ("prev", "thr=0.6"), ("prev", "thr=0.5"), ("prev", "thr=0.85")]}
print(f"{'group':16s} {'claim adds':28s} {'KL':>6s} {'acc':>6s} {'agree':>6s}")
for g, vs in VAR.items():
    for v in vs:
        s = score(X, L, run(X, scrubbed_values(X, [g], opts={g: v}))[0])
        print(f"{g:16s} {'+'.join(v) or '(base)':28s} {s['kl']:6.3f} {s['acc']:6.3f} {s['agree']:6.3f}", flush=True)
