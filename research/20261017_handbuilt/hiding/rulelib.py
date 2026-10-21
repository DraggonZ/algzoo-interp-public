"""A rule for the four hiding neurons (1, 6, 7, 8) of ARC's trained network, and its substitution test.

Exact identity used throughout: pre_n(t) = c_n (ref_n(t) - x_t), c_n = -w_hi[n] (10.56, 11.00, 13.17, 12.31), and
ref_n(t) = (W_hh h(t-1))_n / c_n: the reference is what the previous state makes the neuron compare the arriving number
with. The rule writes c_n ref_n(t) from quantities of the list, in a regime given by t and r = position of the largest
of x_0..x_{t-1} (floor 0; r = -1 while no number is positive); K = t - r is the steps since the record.

Feature groups (all homogeneous of degree 1 in the numbers, since the network has no biases):
  'PM'    previous largest P and largest M (the record)
  'age'   P times an indicator of where P arrived
  'left'  the four's values at the record step (0 after a full wipe; leftover after a partial wipe)
  'since' each number since the record, and the same times whether it was kept (x_j >= 0.75 x the second-largest)
  'clip'  each hiding neuron's clipped amount at each step since the record, e_i(j) = max(-pre_i(j), 0)
  'clip_own' the clipped amounts carried by the four's own block only: (W_cc^(t-j) e(j)), one feature per neuron
  'before' numbers before the record times whether they were kept (memory content), other than P
In the substitution the rule's own clipped amounts and leftovers are used, and everything else runs on its own weights."""
import numpy as np
from common import WI, WH, WO, C, lists, run, kinds

CN = -WI[C]; WCC = WH[np.ix_(C, C)]


def list_vars(X):
    B = len(X); V = {k: np.zeros((B, 10)) for k in ("M", "P", "M2")}; V["R"] = np.full((B, 10), -1); V["RP"] = np.full((B, 10), -1)
    V["KEPT"] = np.zeros((B, 10))
    m = np.zeros(B); p = np.zeros(B); r = np.full(B, -1); rp = np.full(B, -1); m2 = np.zeros(B)
    for t in range(10):
        V["M"][:, t], V["P"][:, t], V["R"][:, t], V["RP"][:, t], V["M2"][:, t] = m, p, r, rp, m2
        x = X[:, t]; new = x > m; V["KEPT"][:, t] = (x >= 0.75 * m2) & (x > 0)
        rp = np.where(new, r, rp); m2 = np.where(new, m, np.maximum(m2, x)); p = np.where(new, m, p)
        m = np.where(new, x, m); r = np.where(new, t, r)
    return V


def features(X, V, t, rr, idx, Hc, E, groups):
    """Hc: the four's values (B, 10, 4); E: clipped amounts (B, 10, 4), both up to step t-1."""
    f = []
    if "PM" in groups: f += [V["P"][idx, t], V["M"][idx, t]]
    if "age" in groups:
        for j in range(-1, max(rr, 0)): f.append(V["P"][idx, t] * (V["RP"][idx, t] == j))
    if "left" in groups and rr >= 0:
        f += list(Hc[idx, rr].T)
    if "since" in groups:
        for j in range(rr + 1, t): f += [X[idx, j], X[idx, j] * V["KEPT"][idx, j]]
    if "clip" in groups:
        for j in range(rr + 1, t): f += list(E[idx, j].T)
    if "clip_own" in groups:
        acc = np.zeros((len(idx), 4))
        for j in range(rr + 1, t): acc += E[idx, j] @ np.linalg.matrix_power(WCC, t - j).T
        f += list(acc.T)
    if "before" in groups:
        for j in range(0, max(rr, 0)): f.append(np.maximum(X[idx, j], 0) * V["KEPT"][idx, j] * (V["RP"][idx, t] != j))
    return np.stack(f, 1) if f else np.zeros((len(idx), 0))


def fit_rule(X, groups, fixed=None):
    """Least squares per (t, r, neuron) of c_n ref_n(t) (minus any fixed part) on the features. Returns coefficients."""
    V = list_vars(X); _, P, H = run(X); Hc = H[:, :, C]; E = np.maximum(-P[:, :, C], 0)
    target = P[:, :, C] - X[:, :, None] * WI[C]
    if fixed is not None: target = target - fixed(X, V, Hc, E)
    co = {}
    for t in range(1, 10):
        for rr in range(-1, t):
            idx = np.where(V["R"][:, t] == rr)[0]
            if len(idx) < 30: continue
            A = features(X, V, t, rr, idx, Hc, E, groups)
            co[(t, rr)] = np.linalg.lstsq(A, target[idx, t], rcond=None)[0] if A.shape[1] else np.zeros((0, 4))
    return co


def substitute(X, co, groups, fixed_step=None, which=None):
    """Run the network with the four (or the subset `which`) replaced by the rule. fixed_step(t, X, V, Hc, E, idx) adds
    a part of c ref not covered by the fitted features (e.g. the four's own block applied to their clipped amounts)."""
    V = list_vars(X); B = len(X); Hc = np.zeros((B, 10, 4)); E = np.zeros((B, 10, 4))
    which = list(range(4)) if which is None else which; cols = [C[i] for i in which]
    def hook(t, pre, hn, hprev):
        cref = np.zeros((B, 4))
        if t > 0:
            for rr in range(-1, t):
                idx = np.where(V["R"][:, t] == rr)[0]
                if len(idx) == 0: continue
                A = features(X, V, t, rr, idx, Hc, E, groups)
                if A.shape[1]: cref[idx] = A @ co[(t, rr)]
                if fixed_step is not None: cref[idx] += fixed_step(t, X, V, Hc, E, idx)
        prer = cref - CN * X[:, t:t + 1]
        hn = hn.copy(); hn[:, cols] = np.maximum(prer[:, which], 0)
        full_pre = pre[:, C].copy(); full_pre[:, which] = prer[:, which]
        Hc[:, t] = hn[:, C]; E[:, t] = np.maximum(-full_pre, 0)
        return hn
    return run(X, hook=hook)


def report(lab, X, pred, base=None, file=None):
    y = X.argsort(1)[:, -2]; K = kinds(X)
    s = f"{lab:50s} " + " | ".join(f"{k.split(' (')[0][:26]} {np.mean(pred[m] == y[m]):.3f}" for k, m in K.items())
    print(s, flush=True)
    return np.mean(pred == y)


def own_echo(t, X, V, Hc, E, idx):
    """The four's own block carrying the leftover at the record step and every clipped amount since:
    W_cc^(t-r) h(r) + sum_j W_cc^(t-j) e(j). Fixed by the weights, nothing fitted."""
    out = np.zeros((len(idx), 4)); R = V["R"][idx, t]
    for rr in np.unique(R):
        sub = R == rr; ii = idx[sub]
        if rr < 0: continue
        acc = Hc[ii, rr] @ np.linalg.matrix_power(WCC, t - rr).T
        for j in range(rr + 1, t): acc += E[ii, j] @ np.linalg.matrix_power(WCC, t - j).T
        out[sub] = acc
    return out


def own_echo_full(X, V, Hc, E):
    B = len(X); out = np.zeros((B, 10, 4))
    for t in range(1, 10):
        out[:, t] = own_echo(t, X, V, Hc, E, np.arange(B))
    return out
