"""Do the comparator neurons' values during the carry explain the copies' two-spike carry losses?

Deterministic θ grid (weights g(θ)Δθ), no sampling. For each copy, after the last spike, overwrite the
comparator neurons (1, 2, 4, 6, 7, 8) with the base model's values on the same input; the copy keeps its own
dense neurons, recurrent weights and readout. Also measure the cancellation in the base model's final margin:
size of the comparator part relative to the margin.
"""
import numpy as np
import torch as th

from algzoo.bench import load_bench
from algzoo.core import example
from algzoo.exact_z2 import HI, LO, g_angle

dev = "cuda"
NG = 1000
tg = LO + (np.arange(NG) + 0.5) * (HI - LO) / NG
W = th.as_tensor(g_angle(tg) * (HI - LO) / NG, device=dev)
cs, sn = th.as_tensor(np.cos(tg), device=dev), th.as_tensor(np.sin(tg), device=dev)
C = [1, 2, 4, 6, 7, 8]
Dn = [i for i in range(16) if i not in C]
T_ = lambda p: tuple(th.as_tensor(w, device=dev, dtype=th.float64) for w in p)


@th.no_grad()
def run(params, p1, p2, override=None):
    w_hi, w_hh, w_oh = params
    h = th.zeros(NG, 16, device=dev, dtype=th.float64)
    hs = []
    for t in range(10):
        x = cs if t == p1 else (sn if t == p2 else th.zeros_like(cs))
        h = th.relu(h @ w_hh.T + x[:, None] * w_hi)
        if override is not None and t > max(p1, p2):
            h = h.clone()
            h[:, C] = override[t][:, C]
        hs.append(h)
    return hs


base = T_(example())
names, acc, models = load_bench(("mult",))
pairs = [(a, b) for a in range(10) for b in range(10) if a != b]
# cancellation in the base model
ratios, wts = [], []
for p1, p2 in pairs:
    hb = run(base, p1, p2)[-1]
    L = hb @ base[2].T
    Lo = L.clone(); Lo[:, p2] = -th.inf
    k = Lo.argmax(-1)
    LC = hb[:, C] @ base[2][:, C].T
    marg = L[:, p2] - L.gather(1, k[:, None])[:, 0]
    cpart = LC[:, p2] - LC.gather(1, k[:, None])[:, 0]
    ok = marg > 0
    ratios.append((cpart.abs() / marg.abs().clamp(min=1e-9))[ok]); wts.append(W[ok])
r, w = th.cat(ratios), th.cat(wts)
o = th.argsort(r); cw = th.cumsum(w[o], 0) / w.sum()
med = r[o][th.searchsorted(cw, th.tensor(0.5, device=dev, dtype=cw.dtype))].item()
print(f"base model, right answers: |comparator part of margin| / |margin|, weighted median {med:.1f}")
rows = []
for nm, m in zip(names, models):
    mp = T_(m)
    lost_own, lost_tr = 0.0, 0.0
    for p1, p2 in pairs:
        T = max(p1, p2)
        hb = run(base, p1, p2)
        hc = run(mp, p1, p2)
        right_T = (hc[T] @ mp[2].T).argmax(-1) == p2 + 9 - T
        okc = (hc[-1] @ mp[2].T).argmax(-1) == p2
        ht = run(mp, p1, p2, override=hb)
        okt = (ht[-1] @ mp[2].T).argmax(-1) == p2
        lost_own += (W * (right_T & ~okc)).sum().item()
        lost_tr += (W * (right_T & ~okt)).sum().item()
    rows.append((float(str(nm).split("_")[0][4:]), lost_own, lost_tr))
R = np.array(rows)
for s in (0.005, 0.01, 0.02):
    k = R[:, 0] == s
    print(f"σ={s}: carry loss {R[k, 1].mean():.4f} -> {R[k, 2].mean():.4f} with the base model's comparator values transplanted")
print(f"all copies: share of carry loss removed {1 - R[:, 2].sum() / R[:, 1].sum():.2f}")
