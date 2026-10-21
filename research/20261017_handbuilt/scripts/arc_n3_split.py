"""ARC neuron 3: direct path (its own output weights) vs indirect path (its weights into
other neurons). Each path gets neuron 3's per-step average while the other path sees the real value."""
import torch as th

from algzoo.core import example

P = [th.tensor(p) for p in example()]
w_hi, w_hh, w_oh = P
g = th.Generator().manual_seed(4)
x = th.randn(1_000_000, 10, generator=g, dtype=th.float64)
tgt = x.argsort(-1)[:, -2]


def run(direct_avg, indirect_avg, mean=None):
    h = th.zeros(len(x), 16, dtype=x.dtype)
    hs = []
    for t in range(10):
        hin = h.clone()
        if indirect_avg and t > 0:
            hin[:, 3] = mean[t - 1]
        h = th.relu(hin @ w_hh.T + x[:, t, None] * w_hi)
        hs.append(h[:, 3].clone())
    hout = h.clone()
    if direct_avg:
        hout[:, 3] = mean[9]
    return (hout @ w_oh.T).argmax(-1), th.stack(hs, 1)


base, n3 = run(False, False)
mean = n3.mean(0)
for lab, d, i in (("direct path averaged", True, False), ("indirect path averaged", False, True), ("both", True, True)):
    pred, _ = run(d, i, mean)
    by = [(pred[tgt == s] == s).double().mean().item() for s in range(10)]
    print(f"{lab:24s} accuracy {(pred == tgt).double().mean():.3f}   by second's slot: " + " ".join(f"{v:.2f}" for v in by))
print(f"{'normal':24s} accuracy {(base == tgt).double().mean():.3f}")
