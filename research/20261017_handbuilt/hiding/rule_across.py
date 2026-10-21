"""Across trained networks with three or more hiding neurons (85% rule): does a rule for the hiding neurons need the
echo of their clipped amounts? Same rule family as rule_ladder.py, per (t, r, neuron) least squares of c ref
(c = -input weight) on: P, M, P's arrival, the hiding neurons' leftover at the record step, numbers since the record
(plain and times 'kept' = at least 0.75 of the second-largest so far) -- 'no echo'; plus each hiding neuron's clipped
amount at each step since the record -- 'with echo'. Substitute the hiding neurons, all others on their own weights.
Fit on 100,000 lists (seed 3), scored on 100,000 (seed 4)."""
import numpy as np, time
from nets import models, hiding


def list_vars(X):
    B = len(X); V = {k: np.zeros((B, 10)) for k in ("M", "P", "M2", "KEPT")}; V["R"] = np.full((B, 10), -1); V["RP"] = np.full((B, 10), -1)
    m = np.zeros(B); p = np.zeros(B); r = np.full(B, -1); rp = np.full(B, -1); m2 = np.zeros(B)
    for t in range(10):
        V["M"][:, t], V["P"][:, t], V["R"][:, t], V["RP"][:, t], V["M2"][:, t] = m, p, r, rp, m2
        x = X[:, t]; new = x > m; V["KEPT"][:, t] = (x >= 0.75 * m2) & (x > 0)
        rp = np.where(new, r, rp); m2 = np.where(new, m, np.maximum(m2, x)); p = np.where(new, m, p); m = np.where(new, x, m); r = np.where(new, t, r)
    return V


def run(params, X, hook=None):
    wi, wh, wo = params; wi = wi.reshape(-1); h = np.zeros((len(X), len(wi))); P, H = [], []
    for t in range(10):
        pre = h @ wh.T + X[:, t:t + 1] * wi; hn = np.maximum(pre, 0)
        if hook is not None: hn = hook(t, pre, hn)
        h = hn; P.append(pre); H.append(h)
    return h @ wo.T, np.stack(P, 1), np.stack(H, 1)


def feats(X, V, t, rr, idx, Hc, E, echo):
    f = [V["P"][idx, t], V["M"][idx, t]] + [V["P"][idx, t] * (V["RP"][idx, t] == j) for j in range(-1, max(rr, 0))]
    if rr >= 0: f += list(Hc[idx, rr].T)
    for j in range(rr + 1, t): f += [X[idx, j], X[idx, j] * V["KEPT"][idx, j]]
    if echo:
        for j in range(rr + 1, t): f += list(E[idx, j].T)
    return np.stack(f, 1)


def fit(params, hid, X, echo):
    wi = params[0].reshape(-1); V = list_vars(X); _, P, H = run(params, X); Hc = H[:, :, hid]; E = np.maximum(-P[:, :, hid], 0)
    T = P[:, :, hid] - X[:, :, None] * wi[hid]; co = {}
    for t in range(1, 10):
        for rr in range(-1, t):
            idx = np.where(V["R"][:, t] == rr)[0]
            if len(idx) < 30: continue
            A = feats(X, V, t, rr, idx, Hc, E, echo); co[(t, rr)] = np.linalg.lstsq(A, T[idx, t], rcond=None)[0]
    return co


def subst(params, hid, X, co, echo):
    wi = params[0].reshape(-1); V = list_vars(X); B = len(X); nh = len(hid); Hc = np.zeros((B, 10, nh)); E = np.zeros((B, 10, nh))
    def hook(t, pre, hn):
        cref = np.zeros((B, nh))
        if t > 0:
            for rr in range(-1, t):
                idx = np.where(V["R"][:, t] == rr)[0]
                if len(idx) == 0 or (t, rr) not in co: continue
                cref[idx] = feats(X, V, t, rr, idx, Hc, E, echo) @ co[(t, rr)]
        prer = cref + wi[hid] * X[:, t:t + 1]; hn = hn.copy(); hn[:, hid] = np.maximum(prer, 0)
        Hc[:, t] = hn[:, hid]; E[:, t] = np.maximum(-prer, 0); return hn
    return run(params, X, hook)[0]


Xf = np.random.default_rng(3).standard_normal((100_000, 10)); Xs = np.random.default_rng(4).standard_normal((100_000, 10)); y = Xs.argsort(1)[:, -2]
for name, params in models().items():
    hid = hiding(params, Xf)
    if len(hid) < 3: continue
    t0 = time.time(); base = np.mean(run(params, Xs)[0].argmax(1) == y); out = []
    for echo in (False, True):
        co = fit(params, hid, Xf, echo); out.append(np.mean(subst(params, hid, Xs, co, echo).argmax(1) == y))
    print(f"{name:7s} hiding {hid}: network {base:.3f} | rule without echo {out[0]:.3f} | with echo {out[1]:.3f}  ({time.time() - t0:.0f} s)", flush=True)
