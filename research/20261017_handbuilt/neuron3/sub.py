"""Q3: substitution tests S1-S6 (predictions in README before running). Fits on lists seed 21 (real run),
tests on 200,000 fresh lists seed 22, closed: the rest of the network reads the substituted neuron 3.
Fits are analysis fits to the network's own activity, never to accuracy."""
from n3lib import *

def quantities(X):
    V = task(X); n, T = X.shape; R = ranks(X)
    srt = np.stack([-np.sort(-np.c_[X[:, :t + 1], np.full((n, 3), 0.0)], 1)[:, :3] for t in range(T)], 1)
    top3sum = np.stack([np.sort(X[:, :t + 1], 1)[:, -3:].sum(1) for t in range(T)], 1)
    rest = np.stack([X[:, :t + 1].sum(1) for t in range(T)], 1) - top3sum
    return V, R, srt, rest

def feats(kind, X, Q, t):
    """features at step t (state after reading x_t)."""
    V, R, srt, rest = Q; n = len(X)
    if kind == "gap": return (srt[:, t, 0] - srt[:, t, 1])[:, None]
    if kind == "m1": return srt[:, t, :1]
    if kind == "four": return np.c_[srt[:, t], rest[:, t]]
    if kind == "none": return np.zeros((n, 0))
    if kind == "pos_rank":  # x_p * 1[rank_p == k], p <= t, k in 1,2,3,4+
        F = np.zeros((n, 10, 4)); rk = np.clip(R[:, t], 1, 4)
        for p in range(t + 1): F[np.arange(n), p, rk[:, p] - 1] = X[:, p]
        return F.reshape(n, 40)
    raise ValueError(kind)

def cells(Q, t): return Q[0]["age"][:, t]

def fit_cells(Y, X, Q, kind, tset, shift):
    """per (t, age) least squares of Y[:, t] on features at step t - shift. Returns dict (t, age) -> coef."""
    C = {}
    for t in tset:
        F = feats(kind, X, Q, t - shift); F = np.c_[F, np.ones(len(X))]; a = cells(Q, t - shift)
        for ag in np.unique(a):
            m = a == ag; c, *_ = np.linalg.lstsq(F[m], Y[m, t], rcond=None); C[(t, ag)] = c
    return C

def apply_cells(C, X, Q, kind, t, shift):
    F = feats(kind, X, Q, t - shift); F = np.c_[F, np.ones(len(X))]; a = cells(Q, t - shift); out = np.zeros(len(X))
    for ag in np.unique(a):
        m = a == ag; out[m] = F[m] @ C.get((t, ag), np.zeros(F.shape[1]))
    return out

def run_mod(X, Q, mode, C=None, kind=None):
    n = len(X); h = np.zeros((n, 16))
    for t in range(10):
        pre = h @ WH.T + X[:, t:t + 1] * WI
        if mode == "D" and t >= 1:      # swap the hiding difference written at t-1
            D = WH[3, 1] * h[:, 1] + WH[3, 8] * h[:, 8]
            pre[:, 3] += apply_cells(C, X, Q, kind, t, 1) - D
        if mode == "pre":
            pre[:, 3] = apply_cells(C, X, Q, kind, t, 0)
        h = np.maximum(pre, 0)
    return h @ WO.T

if __name__ == "__main__":
    Xf, Xt = lists(21, 200_000), lists(22, 200_000)
    Qf, Qt = quantities(Xf), quantities(Xt)
    _, Pf, Hf, *_ = jac(Xf)
    Dreal = np.zeros_like(Pf[:, :, 3]); Dreal[:, 1:] = WH[3, 1] * Hf[:, :-1, 1] + WH[3, 8] * Hf[:, :-1, 8]
    S0 = run_mod(Xt, Qt, "none"); y = Xt.argsort(1)[:, -2]
    def report(name, S):
        ok = S.argmax(1) == y; byp = [ok[y == p].mean() * 100 for p in range(10)]
        print(f"{name:58s} {ok.mean() * 100:5.1f}  by position of the second: " + " ".join(f"{b:5.1f}" for b in byp), flush=True)
    report("network", S0)
    for name, kind in (("S1 hiding difference <- gap formula", "gap"), ("S2 hiding difference <- largest, second, third, rest", "four"),
                       ("S3 hiding difference <- per-cell mean", "none"), ("S4 hiding difference <- largest only", "m1")):
        C = fit_cells(Dreal, Xf, Qf, kind, range(1, 10), 1); report(name, run_mod(Xt, Qt, "D", C, kind))
    for name, kind in (("S5 neuron 3 pre <- position x rank formula", "pos_rank"), ("S6 neuron 3 pre <- rank-only formula", "four")):
        C = fit_cells(Pf[:, :, 3], Xf, Qf, kind, range(10), 0); report(name, run_mod(Xt, Qt, "pre", C, kind))
    C = fit_cells(Pf[:, :, 3], Xf, Qf, "none", range(10), 0); report("control: neuron 3 pre <- per-cell mean", run_mod(Xt, Qt, "pre", C, "none"))
