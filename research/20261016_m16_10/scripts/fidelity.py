"""Fidelity of the pointer operations on clear-cut steps, by age of the stored values.

Pointer = argmax(W_oh h_t), the running 2nd argmax in padded slots (input s at slot s + 10 - t).
For steps where the pointer was right before the step and the new input is far from any tie:
- ordinary: x < 0.8 * running 2nd  -> pointer should shift (stay on the 2nd max)
- new 2nd:  1.2 * 2nd < x < 0.8 * max -> pointer should move to the newest slot
- new max:  x > 1.2 * max -> pointer should move to the old max's slot
Returns P(right after the step) per (event, age of the max, age of the 2nd).
"""
import numpy as np
import torch as th

from core import forward

dev = "cuda"


@th.no_grad()
def fidelity(params, N=2 * 10**6, seed=0, t_min=5):
    w_oh = th.as_tensor(params[2], device=dev, dtype=th.float64)
    x = th.randn(N, 10, device=dev, dtype=th.float64, generator=th.Generator(dev).manual_seed(seed))
    _, H = forward(params, x, return_hidden=True)
    stats = {}
    for t in range(t_min, 10):                 # step consumes x_t, from state after t inputs to t+1
        pre = x[:, :t]
        o = pre.argsort(1, descending=True); v = pre.gather(1, o)
        m1, m2, p1, p2 = v[:, 0], v[:, 1], o[:, 0], o[:, 1]
        ptr0 = (H[:, t] @ w_oh.T).argmax(1); ptr1 = (H[:, t + 1] @ w_oh.T).argmax(1)
        right0 = ptr0 == p2 + (10 - t)
        xt = x[:, t]
        ok_pos = (m2 > 0)
        ev = {
            "ordinary": ok_pos & (xt < 0.8 * m2),
            "new 2nd": ok_pos & (xt > 1.2 * m2) & (xt < 0.8 * m1),
            "new max": ok_pos & (xt > 1.2 * m1),
        }
        want = {"ordinary": p2 + (10 - t - 1), "new 2nd": th.full_like(p2, 9), "new max": p1 + (10 - t - 1)}
        age1 = (t - 1) - p1; age2 = (t - 1) - p2
        for k, sel in ev.items():
            s = sel & right0
            good = (ptr1 == want[k])
            for a1 in range(t):
                for a2 in range(t):
                    ss = s & (age1 == a1) & (age2 == a2)
                    n = int(ss.sum())
                    if n == 0:
                        continue
                    key = (k, a1, a2)
                    c0, n0 = stats.get(key, (0, 0))
                    stats[key] = (c0 + int(good[ss].sum()), n0 + n)
    return stats


def summarize(stats, by="age2"):
    out = {}
    for (k, a1, a2), (c, n) in stats.items():
        key = (k, a2 if by == "age2" else a1)
        c0, n0 = out.get(key, (0, 0))
        out[key] = (c0 + c, n0 + n)
    return {k: (c / n, n) for k, (c, n) in out.items()}
