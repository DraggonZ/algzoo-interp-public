"""r5a analysis 1: per slot of r4a_ring laid out in ARC's order (RMS rule), the correlation with ARC's partner and the
least-squares scale relative to the RMS scale (LS / RMS = corr x RMS(ARC) / RMS(ours) / (RMS(ARC) / RMS(ours)) = the
uncentred correlation). 20,000 lists, seed 5 (not to_arc's seed 0)."""
import numpy as np, torch as th
from algzoo.core import example
from algzoo.handbuilt_v5a import build_v4a, to_arc, acts

x = th.randn(20000, 10, generator=th.Generator().manual_seed(5), dtype=th.float64)
for name, kw in (("r4a_ring", dict(bw_late=True, ring_w=0.062, ring_comp=True)), ("r4a_bw", dict(bw_late=True))):
    W = to_arc(build_v4a(**kw))
    Aa, Am = acts([np.asarray(p, float) for p in example()], x), acts(W, x)
    print(f"== {name}")
    for j in range(16):
        c = np.corrcoef(Am[:, j], Aa[:, j])[0, 1]
        u = (Am[:, j] * Aa[:, j]).sum() / np.sqrt((Am[:, j] ** 2).sum() * (Aa[:, j] ** 2).sum())
        print(f"  slot {j:2d}: corr {c:+.3f}  LS/RMS {u:.3f}  on {100 * (Am[:, j] > 0).mean():5.1f}% (ARC {100 * (Aa[:, j] > 0).mean():5.1f}%)")
