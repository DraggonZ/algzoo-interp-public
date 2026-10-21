"""Review 3 check: z2_attrib.py says restoring the 36 comparator-block weights removes 68% of the copies' two-spike
loss. How much of their full accuracy loss does the same restoration remove? Sampled diagnostic, 2^21 common inputs."""
import numpy as np
import torch as th

from algzoo.bench import load_bench
from algzoo.core import example, forward

C = [1, 2, 4, 6, 7, 8]
g = th.Generator(device="cuda").manual_seed(31)
X = th.randn(2**21, 10, generator=g, device="cuda", dtype=th.float32)
tgt = X.argsort(-1)[:, -2]


@th.no_grad()
def acc(p):
    return (forward(p, X).argmax(-1) == tgt).float().mean().item()


base = example()
a0 = acc(base)
names, true, models = load_bench(("mult",))
sig = np.array([float(str(n).split("_")[0][4:]) for n in names])
res = []
for m in models:
    w_hi, w_hh, w_oh = (np.array(x) for x in m)
    w2 = w_hh.copy()
    w2[np.ix_(C, C)] = base[1][np.ix_(C, C)]
    res.append((acc(m), acc((w_hi, w2, w_oh))))
res = np.array(res)
print(f"base sampled accuracy {a0:.4f}")
for s in (0.005, 0.01, 0.02, None):
    k = np.ones(len(sig), bool) if s is None else sig == s
    lost = (a0 - res[k, 0]).sum()
    back = (res[k, 1] - res[k, 0]).sum()
    print(f"sigma {s or 'all'}: share of full accuracy loss removed by restoring the comparator block {back / lost:.2f}")
