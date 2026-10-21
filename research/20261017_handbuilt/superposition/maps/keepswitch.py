"""Test 4, see README.md."""
import numpy as np, torch as th
from algzoo.core import example, forward
P = [np.asarray(p, float) for p in example()]
rng = np.random.default_rng(11); L = []
while len(L) < 300:
    x = rng.standard_normal(10); v = np.sort(x)[::-1]
    if v[1] > 0 and v[1] - v[2] < 0.05 * v[1]: L.append(x)
S = 401; rows = []
for li, x in enumerate(L):
    o = np.argsort(-x); third = x[o[2]]; g = np.linspace(-3, 0.9 * third, S)
    for p in o[3:]:
        Y = np.repeat(x[None], S, 0); Y[:, p] = g
        a = forward(P, th.tensor(Y)).argmax(-1).numpy(); ch = np.nonzero(np.diff(a))[0]
        if len(ch) != 1: continue
        k = ch[0]; lo, hi = Y[k].copy(), Y[k + 1].copy()
        _, hs = forward(P, th.tensor(np.stack([lo, hi])), return_hidden=True)   # hs[:, t+1] = state after step t
        if p == 9: gate = None                       # no step after the last number
        else: gate = (hs[0, p + 2, 5].item() > 0, hs[1, p + 2, 5].item() > 0)
        prev = x[:p]; m2 = np.sort(prev)[-2] if p >= 2 else np.nan; m1 = prev.max() if p >= 1 else np.nan
        rows.append((li, p, g[k], m2, m1, gate))
n = len(rows); last = sum(r[5] is None for r in rows); G = [r for r in rows if r[5] is not None]
sw = [r for r in G if r[5][0] != r[5][1]]
print(f"one-change sweeps: {n} ({last} of them at position 9, no keep-gate step after)")
print(f"keep gate switches at the flip for {len(sw)} of {len(G)} ({100*len(sw)/len(G):.0f}%)")
for lab, R in (("flips where the keep gate switches", sw), ("other flips", [r for r in G if r[5][0] == r[5][1]])):
    ratio = np.array([r[2] / r[3] for r in R if np.isfinite(r[3]) and r[3] > 0])
    if len(ratio): print(f"  {lab}: flip value / second-largest so far: median {np.median(ratio):.2f}, quartiles {np.percentile(ratio, 25):.2f}-{np.percentile(ratio, 75):.2f}, share in 0.6-0.9: {100*np.mean((ratio >= 0.6) & (ratio <= 0.9)):.0f}% (n {len(ratio)})")
oth = [r for r in G if r[5][0] == r[5][1]]
print("  other flips, keep gate state on both sides (on, off):", sum(r[5][0] for r in oth), sum(not r[5][0] for r in oth))
