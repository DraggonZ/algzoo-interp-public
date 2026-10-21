"""Across the 22 trained networks: does the rest of the network have to see the hiding neurons' clipped values at
partial switches? Hiding neurons: silenced by a new largest (zero at >= 85% of record steps, on at >= 50% of other
steps after step 1; the rule of gao/canc_across.py at 0.85). At steps where some but not all of them are off, every
other neuron receives their unclipped pre-activations instead (the readout still reads the clipped values).
Control: the same change with a random sign per list and hiding neuron.
Networks with fewer than two hiding neurons are skipped (no partial switch possible). 100,000 lists."""
from pathlib import Path
import numpy as np
from algzoo.core import example, zoo, load
models = {"ARC": example()}
for s in range(5): models[f"zoo{s}"] = zoo(16, 10, s)
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["new" + f.stem.split("_")[-1]] = load(str(f))
rng = np.random.default_rng(9); X = rng.standard_normal((100000, 10)); N = len(X); y = np.argsort(X, 1)[:, -2]
SG = np.where(rng.random((N, 16)) < 0.5, -1.0, 1.0)
order = np.argsort(-X, 1); rank = np.argsort(order, 1)
pm = np.maximum.accumulate(np.c_[np.full(N, -np.inf), X], 1)[:, :-1]; rec = X > pm
ar = np.arange(10)[None, :]; recm = rec & (ar >= 1); oth = ~rec & (ar >= 2)
def run(P, hid, mode):
    wi, wh, wo = P; h = np.zeros((N, 16)); hp = h.copy(); hs = []; part_any = np.zeros(N, bool)
    rest = [j for j in range(16) if j not in hid]
    for t in range(10):
        pre = h @ wh.T + X[:, t:t + 1] * wi
        if mode == "fed unclipped":
            pre[:, rest] += (hp[:, hid] - h[:, hid]) @ wh[np.ix_(rest, hid)].T
        if mode == "random signs":             # control: same sizes per neuron, random sign per list and neuron
            pre[:, rest] += (SG[:, :len(hid)] * (hp[:, hid] - h[:, hid])) @ wh[np.ix_(rest, hid)].T
        h = np.maximum(pre, 0); hs.append(h)
        off = (h[:, hid] == 0).sum(1); part = (off > 0) & (off < len(hid)); part_any |= part & (X[:, t] > 0)
        hp = h.copy(); hp[np.ix_(part, hid)] = pre[np.ix_(part, hid)]
    return (h @ wo.T).argmax(1), np.stack(hs, 1), part_any
print(f"{'network':8s} {'hiding':16s} {'lists w/ partial':>16s} {'accuracy':>9s} {'fed unclipped':>14s} {'random signs':>13s}")
for name, w in models.items():
    P = tuple(np.asarray(p, float) for p in w); P = (P[0].reshape(-1), P[1], P[2])
    a0, H, part = run(P, [], "as is")
    hid = [n for n in range(16) if (H[:, :, n][recm] <= 1e-9).mean() >= 0.85 and (H[:, :, n][oth] > 1e-9).mean() >= 0.5]
    if len(hid) < 2:
        print(f"{name:8s} {str(hid):16s} skipped"); continue
    a0, _, part = run(P, hid, "as is"); a1, _, _ = run(P, hid, "fed unclipped"); a2, _, _ = run(P, hid, "random signs")
    print(f"{name:8s} {str(hid):16s} {part.mean():16.2f} {(a0 == y).mean():9.3f} {(a1 == y).mean():14.3f} {(a2 == y).mean():13.3f}", flush=True)
