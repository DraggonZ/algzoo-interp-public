"""Seed 1's extra mistakes (true minus three-input reduction, about 1.4 points): do they depend on the sum of the seven
small inputs? Sampled diagnostic, 2^23 inputs. Mistake rates of the real network and of the reduction (others at their
conditional mean) by quintile of the small inputs' sum, for seed 1 and, for comparison, ARC's model and seed 0."""
import numpy as np
import torch as th

from algzoo.core import example, zoo

dev = "cuda"
N = 2 ** 23
nd = th.distributions.Normal(0.0, 1.0)
g = th.Generator(device=dev).manual_seed(31)
X = th.randn(N, 10, device=dev, generator=g)
o = X.argsort(1); v = X.gather(1, o)
small = v[:, :7].sum(1)
m3 = v[:, -4 + 1]
cm = -th.exp(nd.log_prob(v[:, 7])) / nd.cdf(v[:, 7]).clamp_min(1e-12)
Xr = cm[:, None].expand_as(X).clone(); top3 = o[:, -3:]; Xr.scatter_(1, top3, X.gather(1, top3))
# small-input sum relative to its expectation given the third
dev_sum = small - 7 * cm
q = th.quantile(dev_sum[: 2 ** 20], th.tensor([0.2, 0.4, 0.6, 0.8], device=dev))
bins = th.bucketize(dev_sum, q)
for name in ["example", 0, 1]:
    p = example() if name == "example" else zoo(16, 10, name)
    w_hi, w_hh, w_oh = (th.as_tensor(z, dtype=th.float32, device=dev) for z in p)
    res = []
    for inp in (X, Xr):
        h = th.zeros(N, 16, device=dev)
        for t in range(10):
            h = th.relu(h @ w_hh.T + inp[:, t, None] * w_hi)
        res.append(((h @ w_oh.T).argmax(1) != o[:, -2]).float())
    line = []
    for k in range(5):
        m = bins == k
        line.append(f"q{k + 1}: real {100 * res[0][m].mean():.1f}, reduced {100 * res[1][m].mean():.1f}")
    print(f"{name}: mistake rate (%) by quintile of (small inputs' sum minus its expected value): " + "; ".join(line), flush=True)
