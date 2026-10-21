"""Size-matched control for across.py (C). In each trained network with a late-gate-like neuron (on at 80% or more of
steps right after a record, 10% or less two or more steps after), on near-tie overtakings (by under 5%, q - r = 2..4,
records at 1..5; seed 21, 200,000 lists): accuracy with that neuron zeroed one step after the overtaking, doubled, and,
as control, its value added with a random sign to a random other neuron (not hiding, not late-gate-like) at the same
step (median and range of 5 draws). Also the same on clear overtakings (by 20-50%)."""
import numpy as np
from nets import models, hiding
X = np.random.default_rng(21).standard_normal((200_000, 10)); N = len(X); y = X.argsort(1)[:, -2]
pm = np.maximum.accumulate(np.c_[np.zeros(N), X], 1)[:, :-1]; rec = X > pm
lr = np.full(N, -1); Kst = np.zeros((N, 10), int)
for t in range(10):
    lr = np.where(rec[:, t], t, lr); Kst[:, t] = np.where(lr >= 0, t - lr, -1)


def overtakings(lo, hi):
    sel = np.zeros(N, bool); Q = np.full(N, -1)
    for r_ in range(1, 6):
        for d in (2, 3, 4):
            q_ = r_ + d
            if q_ > 8: continue
            ok = rec[:, r_] & (X[:, r_] > 0.3) & (X[:, r_ + 1:q_].max(1) < X[:, r_]) & (X[:, q_] > (1 + lo) * X[:, r_]) & (X[:, q_] < (1 + hi) * X[:, r_]) & (X[:, q_ + 1:].max(1) < X[:, r_])
            sel |= ok; Q = np.where(ok, q_, Q)
    return sel, Q


def run(params, hook=None):
    wi, wh, wo = params; wi = wi.reshape(-1); h = np.zeros((N, len(wi))); H = []
    for t in range(10):
        pre = h @ wh.T + X[:, t:t + 1] * wi; hn = np.maximum(pre, 0)
        if hook is not None: hn = hook(t, pre, hn)
        h = hn; H.append(h)
    return h @ wo.T, np.stack(H, 1)


rng = np.random.default_rng(5)
for name, params in models().items():
    L0, H0 = run(params); hid = hiding(params, X, H0)
    after = np.zeros((N, 10), bool); after[:, 1:] = rec[:, :-1]
    lg = [n for n in range(16) if n not in hid and (H0[:, :, n][after] > 1e-9).mean() >= 0.8 and (H0[:, :, n][Kst >= 2] > 1e-9).mean() <= 0.1]
    if not lg: continue
    cand = [n for n in range(16) if n not in hid and n not in lg]
    for n in lg:
        out = []
        for lab, lo, hi in (("near tie", 0.0, 0.05), ("clear", 0.2, 0.5)):
            sel, Q = overtakings(lo, hi); ii = np.where(sel)[0]
            def hk(mode, other=None, sign=None):
                def hook(t, pre, hn):
                    m_ = Q == t - 1
                    if not m_.any(): return hn
                    hn = hn.copy()
                    if mode == "zero": hn[m_, n] = 0
                    elif mode == "double": hn[m_, n] *= 2
                    else:
                        idx = np.where(m_)[0]; hn[idx, other[idx]] = np.maximum(hn[idx, other[idx]] + sign[idx] * hn[idx, n], 0)
                    return hn
                return hook
            acc = lambda L: np.mean(L.argmax(1)[ii] == y[ii])
            z = acc(run(params, hk("zero"))[0]); dbl = acc(run(params, hk("double"))[0])
            ctl = [acc(run(params, hk("ctl", rng.choice(cand, N), rng.choice([-1.0, 1.0], N)))[0]) for _ in range(5)]
            out.append(f"{lab} ({len(ii)}): {acc(L0):.3f} -> zeroed {z:.3f}, doubled {dbl:.3f}; size-matched control {np.median(ctl):.3f} ({min(ctl):.3f}-{max(ctl):.3f})")
        print(f"{name:7s} late-gate-like neuron {n}: " + " | ".join(out), flush=True)
