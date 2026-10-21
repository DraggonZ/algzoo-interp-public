"""Behavioural comparison thresholds by age of the stored value.

For each step t (inputs seen so far: t), the pointer argmax(W_oh h_t) is the network's running 2nd argmax
(padded index: input s sits at slot s + 10 - t). We read the network's verdict on the newest input x_{t-1}:

- max comparison: among inputs with x_{t-1} above the running 2nd, "perceived new max" means the pointer
  sits on the old max's slot, "perceived new 2nd" means it sits on the newest slot (9).
- 2nd comparison: among inputs with x_{t-1} below the running max, "perceived new 2nd" means slot 9,
  "perceived lower" means the old 2nd's slot.

Thresholds are reported as the ratio x/m at which the two verdicts are equally likely, per age of the
stored value (steps since it arrived), from a logistic fit on ratios near 1.
"""
import numpy as np
import torch as th

from core import forward

dev = "cuda"


def verdicts(params, N=4 * 10**6, seed=0, t_min=5):
    w_oh = th.as_tensor(params[2], device=dev, dtype=th.float64)
    x = th.randn(N, 10, device=dev, dtype=th.float64, generator=th.Generator(dev).manual_seed(seed))
    _, H = forward(params, x, return_hidden=True)
    rows = {"max": [], "2nd": []}
    for t in range(t_min, 11):           # after t inputs; newest input is x_{t-1}
        pre = x[:, : t - 1]
        o = pre.argsort(1, descending=True)
        v = pre.gather(1, o)
        m1, m2 = v[:, 0], v[:, 1]
        p1, p2 = o[:, 0], o[:, 1]
        xn = x[:, t - 1]
        ptr = (H[:, t] @ w_oh.T).argmax(1)
        slot = lambda s: s + (10 - t)
        age1 = (t - 2) - p1                 # steps since the max arrived, counted at the newest input's arrival
        age2 = (t - 2) - p2
        # max comparison: newest above the running 2nd and near the max
        sel = (xn > m2) & (m1 > 0) & ((xn / m1 - 1).abs() < 0.15)
        newmax_v = (ptr == slot(p1))
        new2_v = (ptr == 9)
        keep = sel & (newmax_v | new2_v)
        rows["max"].append(th.stack([(xn / m1)[keep], newmax_v[keep].double(), age1[keep].double(),
                                     th.full_like(xn[keep], t)], 1))
        # 2nd comparison: newest below the running max and near the running 2nd
        sel = (xn < m1) & (m2 > 0) & ((xn / m2 - 1).abs() < 0.15)
        new2 = (ptr == 9)
        lower = (ptr == slot(p2))
        keep = sel & (new2 | lower)
        rows["2nd"].append(th.stack([(xn / m2)[keep], new2[keep].double(), age2[keep].double(),
                                     th.full_like(xn[keep], t)], 1))
    return {k: th.cat(v).cpu().numpy() for k, v in rows.items()}


def fit_threshold(r, y):
    """Logistic fit P(y=1) = sigmoid((r - c) / w); returns (c, w). Simple Newton on 2 parameters."""
    if len(r) < 200 or y.mean() in (0, 1):
        return np.nan, np.nan
    a, b = 0.0, 10.0                      # logit = b * (r - 1) + a  ->  c = 1 - a/b
    X = np.stack([np.ones_like(r), r - 1], 1)
    beta = np.array([a, b])
    for _ in range(50):
        z = X @ beta
        pz = 1 / (1 + np.exp(-z))
        g = X.T @ (y - pz)
        Hm = -(X * (pz * (1 - pz))[:, None]).T @ X - 1e-6 * np.eye(2)
        step = np.linalg.solve(Hm, g)
        beta = beta - step
        if np.abs(step).max() < 1e-8:
            break
    return 1 - beta[0] / beta[1], 1 / abs(beta[1])


def threshold_table(params, **kw):
    v = verdicts(params, **kw)
    out = {}
    for kind in ("max", "2nd"):
        d = v[kind]
        for age in range(0, 9):
            sel = d[:, 2] == age
            out[(kind, age)] = (*fit_threshold(d[sel, 0], d[sel, 1]), int(sel.sum()))
    return out
