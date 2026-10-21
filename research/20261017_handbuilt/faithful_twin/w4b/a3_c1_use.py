"""w4b analysis read: does the network use what slot 1 (C1) carries? In w4b_margin, C1's push from the late gate is the
only route of the hiding margin (kappa = 0.5 of the jump) into the wave. Here C1 is replaced at every step by C7 (the
same comparator without the push), so the margin never arrives; and, as a control, C6 (slot 6, decides nothing by
design) is replaced by C7. 1,000,000 lists."""
import numpy as np
from algzoo.handbuilt_w4b import build_w4b

rng = np.random.default_rng(6); X = rng.standard_normal((1_000_000, 10)); truth = np.argsort(-X, 1)[:, 1]
wi, wh, wo, idx = build_w4b(cancel=True, margin=True)
def acc(swap=None):
    h = np.zeros((len(X), len(wi)))
    for t in range(10):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
        if swap is not None: h[:, idx[swap]] = h[:, idx["c7"]]
    return 100 * ((h @ wo.T).argmax(1) == truth).mean()
print(f"intact {acc():.2f}%  C1 replaced by C7 (no margin) {acc('c1'):.2f}%  C6 replaced by C7 (control) {acc('c6'):.2f}%")
