"""w7b analysis read 1: does the two-pair wave build (handbuilt_w7b) work, and what does it change against w6b_relay?
Prints: accuracy of both on 400,000 lists, share of lists answered differently, the wave value read at each offset,
the turning modes of the full matrix in the commonest on/off pattern, ARC's slot choice for the 135-degree pair."""
import numpy as np, scipy.linalg as sl
from collections import Counter
from algzoo.handbuilt_w6b import build_w6b
from algzoo.handbuilt_w7b import build_w7b, arc_mode_slot, wave_new


def run(W, x):
    wi, wh, wo = W[:3]; h = np.zeros((len(x), len(wi))); H = []
    for t in range(10):
        h = np.maximum(h @ wh.T + x[:, t:t + 1] * wi, 0); H.append(h)
    return h @ wo.T, np.stack(H, 1)


x = np.random.default_rng(3).standard_normal((400_000, 10))
truth = np.argsort(-x, 1)[:, 1]
W6 = build_w6b(); W7 = build_w7b()
print("neurons", len(W6[0]), len(W7[0]))
s6, H6 = run(W6, x); s7, H7 = run(W7, x)
a6, a7 = s6.argmax(1), s7.argmax(1)
print(f"accuracy w6b {100 * (a6 == truth).mean():.2f}  w7b {100 * (a7 == truth).mean():.2f}  differ {100 * (a6 != a7).mean():.2f}%")
print("wave f(k), k=-9..9:", np.round(wave_new(np.arange(-9, 10)), 2))
# errors by where the largest is, for both
pos1 = x.argmax(1)
for name, a_ in (("w6b", a6), ("w7b", a7)):
    print(name, "error by largest's position:", np.round([100 * (a_[pos1 == p] != truth[pos1 == p]).mean() for p in range(10)], 1))
    print(name, "wrong answer = largest:", f"{100 * (a_ == pos1).mean():.2f}%")
for name, W, H in (("w6b", W6, H6), ("w7b", W7, H7)):
    on = (H[:, 4:9] > 0).reshape(-1, H.shape[2])
    top = np.frombuffer(Counter(map(lambda q: q.tobytes(), on[:50000])).most_common(1)[0][0], dtype=bool)
    ev = sl.eigvals(np.diag(top.astype(float)) @ W[1])
    print(name, "turning modes (deg, modulus):", [(round(float(np.degrees(np.angle(e))), 1), round(abs(e), 3)) for e in ev if e.imag > 1e-9 and abs(e) > 0.3])
j, deg, mod, share = arc_mode_slot()
print(f"ARC's mode nearest 135: {deg:.1f} deg, modulus {mod:.2f}; share per neuron {np.round(share, 2)}; chosen slot {j}")
