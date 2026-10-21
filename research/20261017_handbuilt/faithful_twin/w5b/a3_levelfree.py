"""w5b analysis read: the level-free rule (handbuilt_w5b). (1) Same function as w4b_margin? answers on 400,000 lists.
(2) The frozen version-2 hiding test's function with w5b's map (slot 7 = the four wave neurons, C7 a helper), and with
other silence sets for attribution."""
import sys
import numpy as np
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/faithful_twin")
from score_wide import Net  # noqa: E402
from score_mechanism2 import names_largest_when_silenced  # noqa: E402
from algzoo.handbuilt_w4b import build_w4b
from algzoo.handbuilt_w5b import build_w5b, w5b_map

old = build_w4b(cancel=True, margin=True); new = build_w5b()
rng = np.random.default_rng(3); X = rng.standard_normal((400_000, 10)); truth = np.argsort(-X, 1)[:, 1]
def scores(W):
    wi, wh, wo = W[:3]; h = np.zeros((len(X), len(wi)))
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h @ wo.T
so, sn = scores(old), scores(new)
print(f"answers changed {100 * (so.argmax(1) != sn.argmax(1)).mean():.3f}%; accuracy old {100 * (so.argmax(1) == truth).mean():.2f}, new {100 * (sn.argmax(1) == truth).mean():.2f}; max score change {np.abs(so - sn).max():.2e}")
M, lay = w5b_map(*new)
print("layout:", {k: lay[k] for k in sorted(lay)})
wi, wh, wo, idx = new
print(f"w5b map (hiding rows: {lay[1]}, {lay[6]}, {lay[7]}, {lay[8]}): names the largest {100 * names_largest_when_silenced(Net([wi, wh, wo], M)):.1f}%", flush=True)
n = len(wi)
for name, ks in (("wave only", ["clock0", "clock1", "clock2", "flip"]), ("wave + Q", ["clock0", "clock1", "clock2", "flip", "Q"]),
                 ("wave + c1 + c6", ["clock0", "clock1", "clock2", "flip", "c1", "c6"])):
    M2 = np.zeros((16, n)); rows = [1, 6, 7, 8]
    for j, k in enumerate(ks): M2[rows[j % 4], idx[k]] = 1.0
    free = [r for r in range(16) if r not in rows]
    for r, k in zip(free, [k for k in range(n) if M2[:, k].sum() == 0]): M2[r, k] = 1.0
    print(f"{name:20s} names the largest {100 * names_largest_when_silenced(Net([wi, wh, wo], M2)):.1f}%", flush=True)
