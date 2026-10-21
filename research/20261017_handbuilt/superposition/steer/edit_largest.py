"""Edit 1: turn the network into a largest-finder, designed from the account. Predictions written before running:
(a) Silencing the hiding neurons at every step: the corrected account says the memory and store share the hiding
    after a few steps, so the largest is only partly revealed: predicted largest named on roughly 60-80% of lists.
(b) Read the hiding neurons alone: they push minus the largest's wave, peaked at its position, so the position with
    the lowest hiding readout should be the largest: predicted right on 90-99%; failures mostly when the largest
    arrives at the last step (the detector acts a step later), about 10% of lists, so nearer 90%.
(c) Also the store's discount and memory imprint: no simple edit; not tried."""
from harness import *
X = lists(1_000_000, seed=2); L = X.argmax(1)
S = run(ARC, X, [(GROUPS["hiding"], set(range(10)), 0.0)])
print(f"(a) hiding silenced at every step: names the largest {100*(S.argmax(1) == L).double().mean().item():.1f}%, the second {100*(S.argmax(1) == X.argsort(1, descending=True)[:, 1]).double().mean().item():.1f}%")
_, Hs = run(ARC, X, keep=True); hid = Hs[-1][:, GROUPS["hiding"]] @ ARC[2][:, GROUPS["hiding"]].T
a = hid.argmin(1); ok = a == L
print(f"(b) lowest hiding readout names the largest {100*ok.double().mean().item():.1f}%")
for p in range(10):
    m = L == p; print(f"     largest at step {p}: {100*ok[m].double().mean().item():5.1f}%")
