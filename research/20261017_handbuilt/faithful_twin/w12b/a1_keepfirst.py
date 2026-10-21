"""w12b analysis 1: w8b with the keep gate as first vote cell (linear route without readers). Own accuracy, overall and
by the position of the second-largest and of the largest; 1M lists."""
import numpy as np
from algzoo.handbuilt_w8b import build_w8b
from algzoo.handbuilt_w12b import free_linear


def run(wi, wh, wo, x):
    h = np.zeros((len(x), len(wi)))
    for t in range(10):
        h = np.maximum(h @ wh.T + x[:, t:t + 1] * wi, 0)
    return np.argmax(h @ wo.T, 1)


x = np.random.default_rng(1).standard_normal((1_000_000, 10))
o = np.argsort(x, 1); sec, top = o[:, -2], o[:, -1]
wi, wh, wo, idx, _ = build_w8b()
nets = {"w8b": (wi, wh, wo)}
a, b, c, info = free_linear(wi, wh, wo, idx); nets["keep-first"] = (a, b, c)
print("solved: r =", info["r"], " lam =", info["lam"], " back on clocks =", np.round(info["back"], 4))
a, b, c, _ = free_linear(wi, wh, wo, idx, pos8=False); nets["keep-first, readout pos 8 still reads L"] = (a, b, c)
for nm, W in nets.items():
    p = run(*W, x); ok = p == sec
    print(f"\n{nm}: accuracy {100 * ok.mean():.2f}  names largest {100 * (p == top).mean():.2f}")
    print("  by second's position:", " ".join(f"{100 * ok[sec == j].mean():5.1f}" for j in range(10)))
    print("  by largest's position:", " ".join(f"{100 * ok[top == j].mean():5.1f}" for j in range(10)))
    print("  wrong answers' position:", np.bincount(p[~ok], minlength=10))
