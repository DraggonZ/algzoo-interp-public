"""Across the 22 trained networks (nets.py), for those with hiding neurons (85% rule):
(A) undoing partial clipping of the hiding neurons, by step since the record (K = 0 record step, K = 1, K >= 2), as
    partial_by_step.py; accuracy overall and on lists whose answer sits right after the largest; control: the same for
    as many random other neurons on the same steps (median of 3 draws).
(B) do the hiding neurons' references follow the running second-largest m2? Per hiding neuron, its reference times c
    (pre-activation + c x, c = -input weight) at K = 2 and 3 steps after a record, fitted on (P, M) and on (m2, M),
    pooled over records at 1..6 with P > 0.2; residual RMS ratio (m2, M) / (P, M), and the share of steps where m2 > P.
(C) a late-gate-like neuron: on at 80% or more of steps right after a record and at 10% or less of steps two or more
    after; zeroed one step after a near-tie overtaking (by under 5%, q - r = 2..4): accuracy against zeroing a random
    other neuron (not hiding, not late-gate-like; median of 5 draws).
200,000 lists, seed 21."""
import numpy as np
from nets import models, hiding
X = np.random.default_rng(21).standard_normal((200_000, 10)); N = len(X); y = X.argsort(1)[:, -2]; o = np.argsort(-X, 1)
lar, sec = o[:, 0], o[:, 1]
pm = np.maximum.accumulate(np.c_[np.zeros(N), X], 1)[:, :-1]; rec = X > pm
lr = np.full(N, -1); Kst = np.zeros((N, 10), int); M1 = np.zeros((N, 10)); Pp = np.zeros((N, 10)); M2 = np.zeros((N, 10))
m = np.zeros(N); p = np.zeros(N); m2 = np.zeros(N)
for t in range(10):
    M1[:, t], Pp[:, t], M2[:, t] = m, p, m2      # before step t
    new = X[:, t] > m; m2 = np.where(new, m, np.maximum(m2, X[:, t])); p = np.where(new, m, p); m = np.where(new, X[:, t], m)
    lr = np.where(rec[:, t], t, lr); Kst[:, t] = np.where(lr >= 0, t - lr, -1)
R = np.zeros((N, 10), int); rr_ = np.full(N, -1)
for t in range(10):
    R[:, t] = rr_; rr_ = np.where(rec[:, t], t, rr_)
right_after = sec == lar + 1
rng = np.random.default_rng(0)


def run(params, hook=None):
    wi, wh, wo = params; wi = wi.reshape(-1); h = np.zeros((N, len(wi))); P, H = [], []
    for t in range(10):
        pre = h @ wh.T + X[:, t:t + 1] * wi; hn = np.maximum(pre, 0)
        if hook is not None: hn = hook(t, pre, hn)
        h = hn; P.append(pre); H.append(h)
    return h @ wo.T, np.stack(P, 1), np.stack(H, 1)


def undo(params, hid, group, kcls):
    def hook(t, pre, hn):
        off = (pre[:, hid] <= 0).sum(1); part = (off > 0) & (off < len(hid)); k = Kst[:, t]
        sel = part & {"K0": k == 0, "K1": k == 1, "K2+": k >= 2}[kcls]
        hn = hn.copy(); hn[np.ix_(sel, group)] = pre[np.ix_(sel, group)]; return hn
    return run(params, hook)[0].argmax(1)


# near-tie overtakings for (C)
nt = np.zeros(N, bool); Q = np.full(N, -1)
for r_ in range(1, 6):
    for d in (2, 3, 4):
        q_ = r_ + d
        if q_ > 8: continue
        ok = rec[:, r_] & (X[:, r_] > 0.3) & (X[:, r_ + 1:q_].max(1) < X[:, r_]) & (X[:, q_] > X[:, r_]) & (X[:, q_] < 1.05 * X[:, r_]) & (X[:, q_ + 1:].max(1) < X[:, r_])
        nt |= ok; Q = np.where(ok, q_, Q)

for name, params in models().items():
    L0, P0, H0 = run(params); acc0 = np.mean(L0.argmax(1) == y); hid = hiding(params, X, H0)
    print(f"== {name}: accuracy {acc0:.3f}; hiding neurons {hid}", flush=True)
    if hid:
        others = [n for n in range(16) if n not in hid]
        for kcls in ("K0", "K1", "K2+"):
            pr = undo(params, hid, hid, kcls)
            ctl = [np.mean(undo(params, hid, list(rng.choice(others, min(len(hid), len(others)), replace=False)), kcls) == y) for _ in range(3)]
            print(f"   (A) undo at {kcls:3s}: accuracy {np.mean(pr == y):.3f} (answer right after the largest {np.mean(pr[right_after] == y[right_after]):.3f}, normally {np.mean(L0.argmax(1)[right_after] == y[right_after]):.3f}); random other neurons {np.median(ctl):.3f}", flush=True)
        wi = params[0].reshape(-1)
        for n in hid:
            cn = -wi[n]
            if cn <= 0: print(f"   (B) neuron {n}: input weight not negative ({wi[n]:+.2f}), skipped"); continue
            out = []
            for K in (2, 3):
                rows = []
                for r_ in range(1, 7):
                    t = r_ + K
                    if t > 9: continue
                    idx = np.where((R[:, t] == r_) & (Pp[:, t] > 0.2))[0]
                    rows.append(np.c_[Pp[idx, t], M2[idx, t], M1[idx, t], P0[idx, t, n] + cn * X[idx, t]])
                A = np.concatenate(rows); yv = A[:, 3]
                r1 = np.sqrt(np.mean((yv - A[:, [0, 2]] @ np.linalg.lstsq(A[:, [0, 2]], yv, rcond=None)[0]) ** 2))
                r2 = np.sqrt(np.mean((yv - A[:, [1, 2]] @ np.linalg.lstsq(A[:, [1, 2]], yv, rcond=None)[0]) ** 2))
                co = np.linalg.lstsq(A[:, [1, 2]], yv, rcond=None)[0] / cn
                out.append(f"K={K}: rms ratio {r2 / r1:.2f} (m2 weight {co[0]:+.2f}, M {co[1]:+.2f}; m2 > P on {np.mean(A[:, 1] > A[:, 0] + 1e-9):.2f})")
            print(f"   (B) neuron {n}: " + " | ".join(out), flush=True)
    # (C)
    after = np.zeros((N, 10), bool); after[:, 1:] = rec[:, :-1] & (np.arange(9)[None] >= 0)
    later = (Kst >= 2)
    lg = [n for n in range(16) if n not in hid and (H0[:, :, n][after] > 1e-9).mean() >= 0.8 and (H0[:, :, n][later] > 1e-9).mean() <= 0.1]
    if not lg: print("   (C) no late-gate-like neuron"); continue
    ii = np.where(nt)[0]; ynt = y[ii]
    def zero_at(nlist):
        def hook(t, pre, hn):
            m_ = Q == t - 1
            if m_.any(): hn = hn.copy(); hn[np.ix_(m_, nlist)] = 0
            return hn
        return run(params, hook)[0].argmax(1)
    base = np.mean(L0.argmax(1)[ii] == ynt)
    cand = [n for n in range(16) if n not in hid and n not in lg]
    for n in lg:
        pz = zero_at([n]); ctl = [np.mean(zero_at([int(rng.choice(cand))])[ii] == ynt) for _ in range(5)]
        rates = ((H0[:, :, n][after] > 1e-9).mean(), (H0[:, :, n][later] > 1e-9).mean())
        print(f"   (C) late-gate-like neuron {n} (on {rates[0]:.2f} right after a record, {rates[1]:.2f} later): near-tie overtakings ({len(ii)} lists) accuracy {base:.3f} -> zeroed {np.mean(pz[ii] == ynt):.3f}; random other neuron zeroed {np.median(ctl):.3f} (range {min(ctl):.3f}-{max(ctl):.3f})", flush=True)
