"""Substitution ladder. The four hiding neurons are replaced at every step by ReLU(W_cc c(t-1) + d(t) + w x_t), with their
own trained 4x4 block W_cc and a drive d(t) from the rest of the network, all other neurons on their own weights (they
see the substituted values). Variants of d: the true drive (sanity, = network); a drive fitted per (neuron, r, K) on
input features (previous largest P with where it arrived, largest M, numbers since the record, and those numbers times
whether the keep gate kept them); the same but with the store's (or the store's and detector's) term read from the
running network. Fit lists seed 3, scored on seed 4 (200,000 each). Accuracy overall and by kind of list."""
import numpy as np
from common import WI, WH, WO, C, GROUPS, lists, run, kinds
Wcc = WH[np.ix_(C, C)]; REST = [n for n in range(16) if n not in C]


def state_vars(X):
    B = len(X); M = np.zeros((B, 10)); Pp = np.zeros((B, 10)); R = np.full((B, 10), -1); RP = np.full((B, 10), -1)
    KEPT = np.zeros((B, 10)); m = np.zeros(B); p = np.zeros(B); r = np.full(B, -1); rp = np.full(B, -1); m2 = np.zeros(B)
    for t in range(10):
        M[:, t], Pp[:, t], R[:, t], RP[:, t] = m, p, r, rp
        x = X[:, t]; new = x > m; KEPT[:, t] = (x >= 0.75 * m2) & (x > 0)
        rp = np.where(new, r, rp); m2 = np.where(new, m, np.maximum(m2, x)); p = np.where(new, m, p)
        m = np.where(new, x, m); r = np.where(new, t, r)
    return M, Pp, R, RP, KEPT


def feats(X, V, t, rr, idx):
    M, Pp, R, RP, KEPT = V; f = [Pp[idx, t], M[idx, t]]
    for j in range(rr + 1, t): f += [X[idx, j], np.maximum(X[idx, j], 0), X[idx, j] * KEPT[idx, j]]
    for j in range(-1, max(rr, 0)): f += [Pp[idx, t] * (RP[idx, t] == j)]
    return np.stack(f, 1)


def fit(X, sources):
    """Per (r, t): coefficients of the drive from the sources NOT read live."""
    V = state_vars(X); _, P, H = run(X); Hp = np.concatenate([np.zeros_like(H[:, :1]), H[:, :-1]], 1)
    idx_src = [n for n in REST if n not in sources]
    drive = Hp[:, :, idx_src] @ WH[np.ix_(C, idx_src)].T
    co = {}
    for t in range(1, 10):
        for rr in range(-1, t):
            idx = np.where(V[2][:, t] == rr)[0]
            if len(idx) < 50: continue
            A = feats(X, V, t, rr, idx); co[(t, rr)] = np.linalg.lstsq(A, drive[idx, t], rcond=None)[0]
    return co


def subst(X, co, sources):
    V = state_vars(X); B = len(X)
    def hook(t, pre, hn, hprev):
        if t == 0: return hn
        d = hprev[:, sources] @ WH[np.ix_(C, sources)].T if sources else np.zeros((B, 4))
        for rr in range(-1, t):
            idx = np.where(V[2][:, t] == rr)[0]
            if len(idx) == 0: continue
            d[idx] += feats(X, V, t, rr, idx) @ co[(t, rr)]
        hn = hn.copy(); hn[:, C] = np.maximum(hprev[:, C] @ Wcc.T + d + X[:, t:t + 1] * WI[C], 0); return hn
    return run(X, hook=hook)[0]


Xf, Xs = lists(3, 200_000), lists(4, 200_000); y = Xs.argsort(1)[:, -2]; K = kinds(Xs)
base = run(Xs)[0].argmax(1)
def report(lab, pred):
    print(f"{lab:58s} all {np.mean(pred == y):.3f} | " + " | ".join(f"{k[:28]} {np.mean(pred[m] == y[m]):.3f}" for k, m in K.items() if k != "all"), flush=True)
report("network", base)
for lab, src in (("drive fitted (all sources)", []), ("store read live, rest fitted", [2]),
                 ("store and detector live, rest fitted", [2, 4]), ("store, detector, gates live, memory fitted", [2, 4, 5, 9]),
                 ("all live (sanity)", REST)):
    co = fit(Xf, src); report(lab, subst(Xs, co, src).argmax(1))
