"""w4b analysis read: do the extra neurons leave r3b's function unchanged, and do they hold what the design says?
200,000 lists. Prints accuracy of each build, the share of lists whose answer differs from r3b's network, and checks
LG = d one step after a record, C7 = g relu(m1 - x), C6 / C1 references."""
import numpy as np
from algzoo.handbuilt_w4b import build_w4b
from algzoo.handbuilt_v27b import build_v27b

rng = np.random.default_rng(3); X = rng.standard_normal((200_000, 10))
def run(wi, wh, wo):
    h = np.zeros((len(X), len(wi))); H = []
    for t in range(10): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); H.append(h)
    return np.stack(H, 1), h @ wo.T
truth = np.argsort(-X, 1)[:, 1]
base = [np.asarray(p, float) for p in build_v27b(level_exact=True).params]
_, Sb = run(*base); ab = Sb.argmax(1)
print(f"r3b (v27b level_exact): accuracy {100 * (ab == truth).mean():.2f}")
for opts in (dict(late=True, level=False, refs=False), dict(late=True, level=True, refs=False), dict()):
    wi, wh, wo, idx = build_w4b(**opts); H, Sc = run(wi, wh, wo); ac = Sc.argmax(1)
    print(f"{opts or 'full'}: n={len(wi)} accuracy {100 * (ac == truth).mean():.2f}, answers differing from r3b {100 * (ac != ab).mean():.3f}%, "
          f"largest score difference {np.abs(Sc - Sb).max():.2e}")
wi, wh, wo, idx = build_w4b(); H, _ = run(wi, wh, wo)
M = np.maximum.accumulate(X, 1); m_prev = np.c_[np.full(len(X), -np.inf), M[:, :-1]]
rec = X > m_prev; rec[:, 0] = X[:, 0] > 0   # first step: the store starts at 0
d = np.maximum(X - np.c_[np.zeros(len(X)), np.maximum(M[:, :-1], 0)], 0)
LG = H[:, 1:, idx["late"]]; print("LG vs d one step earlier: max error", np.abs(LG - d[:, :-1]).max())
m1 = np.maximum(M, 0)
C7 = H[:, 1:, idx["c7"]]; print("C7 vs g relu(m1(t-1) - x_t): max rel error", np.abs(C7 - 1e4 * np.maximum(m1[:, :-1] - X[:, 1:], 0)).max() / 1e4)
for k in ("c7", "c6", "c1", "Q", "late"):
    h = H[:, 1:, idx[k]]; print(f"{k}: on after records {(h[rec[:, 1:]] > 0).mean():.3f}, on elsewhere {(h[~rec[:, 1:]] > 0).mean():.3f}")
