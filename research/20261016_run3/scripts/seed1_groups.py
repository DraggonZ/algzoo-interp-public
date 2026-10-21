"""Which small inputs cause seed 1's excess mistakes? (review 6 direction 3; kill rule: no group holds more than half)

Start from the reduced input (top three exact, the other seven at E[X | X < third]). Restore one group of small inputs to
their true values and measure the mistake rate. Groups: arriving before the first of the top three, between them, after
the last; positive or negative; the fourth-largest alone. Excess = mistakes(full) - mistakes(reduced). Sampled, 2^22."""
import numpy as np
import torch as th

from algzoo.core import example, zoo

dev = "cuda"
N = 2 ** 22
nd = th.distributions.Normal(0.0, 1.0)
g = th.Generator(device=dev).manual_seed(161)
X = th.randn(N, 10, device=dev, generator=g)
o = X.argsort(1); tgt = o[:, -2]
rank = o.argsort(1)                       # rank of each position (0 smallest)
m3 = X.gather(1, o[:, -3:-2])[:, 0]
cm = -th.exp(nd.log_prob(m3)) / nd.cdf(m3).clamp_min(1e-12)
small = rank < 7
top = ~small
pos = th.arange(10, device=dev)[None, :].expand(N, 10)
first = th.where(top, pos, 10).min(1).values; last = th.where(top, pos, -1).max(1).values
groups = {"before the top three": small & (pos < first[:, None]),
          "between them": small & (pos > first[:, None]) & (pos < last[:, None]),
          "after the last": small & (pos > last[:, None]),
          "positive": small & (X > 0), "negative": small & (X <= 0),
          "fourth-largest alone": rank == 6}
Xr = th.where(small, cm[:, None], X)


def wrong(p, inp):
    w_hi, w_hh, w_oh = (th.as_tensor(np.asarray(q), dtype=th.float32, device=dev) for q in p)
    h = th.zeros(N, 16, device=dev)
    for t in range(10):
        h = th.relu(h @ w_hh.T + inp[:, t, None] * w_hi)
    return 100 * ((h @ w_oh.T).argmax(1) != tgt).float().mean().item()


for name in ["seed1", "example", "seed0"]:
    p = example() if name == "example" else zoo(16, 10, int(name[-1]))
    wf, wr = wrong(p, X), wrong(p, Xr)
    line = [f"{name}: mistakes full {wf:.2f}, reduced {wr:.2f}, excess {wf - wr:+.2f} points"]
    for lab, gm in groups.items():
        wg = wrong(p, th.where(gm, X, Xr))
        line.append(f"restore {lab}: {wg - wr:+.2f}")
    print("; ".join(line), flush=True)
