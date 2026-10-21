"""Check the reduction's mistake map (mistakes_map.py) against the real network on sampled full inputs (2^23 per model)."""
import torch as th

from algzoo.core import example, zoo

dev = "cuda"
N, CH = 2 ** 23, 2 ** 22
for name in ["example", 2, 3, 0, 4, 1]:
    p = example() if name == "example" else zoo(16, 10, name)
    w_hi, w_hh, w_oh = (th.as_tensor(q, dtype=th.float32, device=dev) for q in p)
    g = th.Generator(device=dev).manual_seed(99)
    acc = {k: 0.0 for k in ["second <= 0", "largest-second < 0.1 only", "second-third < 0.1 only", "both gaps < 0.1", "neither"]}
    pick3 = {k: 0.0 for k in acc}; tot = 0.0
    for _ in range(N // CH):
        X = th.randn(CH, 10, device=dev, generator=g)
        h = th.zeros(CH, 16, device=dev)
        for t in range(10):
            h = th.relu(h @ w_hh.T + X[:, t, None] * w_hi)
        pred = (h @ w_oh.T).argmax(1)
        o = X.argsort(1); v = X.gather(1, o)
        wrong = pred != o[:, -2]
        m1, m2, m3 = v[:, -1], v[:, -2], v[:, -3]
        neg = m2 <= 0; n12 = (m1 - m2 < 0.1) & ~neg; n23 = (m2 - m3 < 0.1) & ~neg
        kinds = {"second <= 0": neg, "largest-second < 0.1 only": n12 & ~n23, "second-third < 0.1 only": n23 & ~n12,
                 "both gaps < 0.1": n12 & n23, "neither": ~neg & ~n12 & ~n23}
        for k, m in kinds.items():
            acc[k] += (wrong & m).sum().item(); pick3[k] += (wrong & m & (pred == o[:, -3])).sum().item()
        tot += wrong.sum().item()
    print(f"{name}: mistakes {100 * tot / N:.2f} points; " + "; ".join(f"{k} {100 * v / N:.2f} (third {pick3[k] / max(v, 1):.0%})" for k, v in acc.items()), flush=True)
