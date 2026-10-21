"""The cheap tie-band estimator (deterministic, no random inputs).

Three aligned nodes, all with the seven smaller numbers at -0.40 (the mean background near the tie planes' centroid):
  2nd-3rd tie  (m1, m2, m3) = (u0 + b0, u0, u0)          one run per unordered tied pair + largest's position (360)
  1st-2nd tie  (u0 + a0, u0 + a0, u0)                    one run per tied pair + third's position (360)
  bulk         (u0 + a0 + b0, u0 + a0, u0)               one run per ordered triple, only where needed
with u0 = 0.78, a0 = 0.37, b0 = 0.58 (the tie planes' density-weighted centroids). Each tie run gives the margin D
(true second's score minus the best other) for both orderings that share it. Per ordered triple:
  right at a tie  -> no band at that tie;
  wrong at a tie  -> band width at that tie = the zero of the margin on the straight line from the tie (margin D)
                     to the bulk node (margin M) when the bulk run exists and M > 0: width = gap * (-D) / (M - D);
                     if M <= 0 the triple is wrong far from the tie too: the band is 'wide' (width WIDE);
                     without a bulk run: width = -D / slope, slope = median of the straight-line slopes (M - D) / gap
                     from the triples that have bulk runs (one per tie type).
Then bandlib.accuracy integrates 'm2 < 0, or a < A S23, or b < B S12' with the exact top-three density (no runs).
mode 'double': bulk runs only for triples wrong at both ties; 'losers': for triples wrong at either; 'all': all 720.
Cost: runs (distinct 10-number inputs) and RNN-step equivalents (steps / 10, with inputs sharing a prefix computed once)."""
import numpy as np, torch
import common as c, bandlib as bl

U0, A0, B0, BG0 = 0.78, 0.37, 0.58, -0.40
WIDE = 5.0
N23 = (U0 + B0, U0, U0, BG0); N12 = (U0 + A0, U0 + A0, U0, BG0); NB = (U0 + A0 + B0, U0 + A0, U0, BG0)


def _X(P, nd):
    n = len(P); ar = torch.arange(n, device=c.DEV); X = torch.full((n, 10), nd[3], device=c.DEV, dtype=c.dt)
    for k in range(3): X[ar, P[:, k]] = nd[k]
    return X


def step_cost(Xs):
    """Distinct prefixes over all inputs = RNN steps when shared prefixes are computed once; plus readouts."""
    X = torch.cat(Xs).cpu().numpy(); pre = set()
    for row in X:
        t = tuple(np.round(row, 9))
        for k in range(1, 11): pre.add(t[:k])
    return len(pre) / 10


def estimate(W, mode="double", law="norm", return_parts=False, slope_rule="secant", n_tan=None):
    T = c.T; n = 720; Tn = T.cpu().numpy()
    idx = {tuple(t): i for i, t in enumerate(Tn.tolist())}
    # tie runs: one per unordered tie configuration
    keys23 = sorted({(p1, min(p2, p3), max(p2, p3)) for p1, p2, p3 in Tn.tolist()})
    keys12 = sorted({(min(p1, p2), max(p1, p2), p3) for p1, p2, p3 in Tn.tolist()})
    P23 = torch.tensor([(k[0], k[1], k[2]) for k in keys23], device=c.DEV)   # (largest, tied, tied)
    P12 = torch.tensor([(k[0], k[1], k[2]) for k in keys12], device=c.DEV)   # (tied, tied, third)
    X23 = _X(P23, N23); X12 = _X(P12, N12)
    S23, _ = c.scores(W, X23); S12, _ = c.scores(W, X12)
    D = {"23": np.zeros(n), "12": np.zeros(n)}
    S23n, S12n = S23.cpu().numpy(), S12.cpu().numpy()
    for i, (p1, p2, p3) in enumerate(Tn.tolist()):
        s = S23n[keys23.index((p1, min(p2, p3), max(p2, p3)))]; D["23"][i] = s[p2] - np.delete(s, p2).max()
        s = S12n[keys12.index((min(p1, p2), max(p1, p2), p3))]; D["12"][i] = s[p2] - np.delete(s, p2).max()
    los = {t: D[t] < 0 for t in D}
    need = {"double": los["23"] & los["12"], "losers": los["23"] | los["12"], "all": np.ones(n, bool)}[mode]
    Xb = _X(T[torch.tensor(need, device=c.DEV)], NB)
    Sb, _ = c.scores(W, Xb); Mb = np.full(n, np.nan)
    Sbn = Sb.cpu().numpy(); Mb[need] = Sbn[np.arange(need.sum()), Tn[need, 1]] - np.array([np.delete(s, p).max() for s, p in zip(Sbn, Tn[need, 1])])
    widths = {}; Xt = []
    if slope_rule == "tangent":
        # forward tangents along each tie's gap at the tie inputs of losers (every n_tan-th loser, or all)
        tan = {}
        for t, Xtie, keys, nd in (("23", X23, keys23, N23), ("12", X12, keys12, N12)):
            li = np.where(los[t])[0]
            if n_tan: li = li[np.linspace(0, len(li) - 1, n_tan).round().astype(int)]
            Pl = T[torch.tensor(li, device=c.DEV)]; Xl = _X(Pl, nd); dX = torch.zeros_like(Xl); ar = torch.arange(len(li), device=c.DEV)
            dX[ar, Pl[:, 0]] = 1
            if t == "23": dX[ar, Pl[:, 1]] = 1
            S_, dS = c.scores(W, Xl, dX); _, sl, _ = c.margin_from(S_, Pl[:, 1], dS)
            tan[t] = (li, sl.cpu().numpy()); Xt.append(Xl)
    for t, gap in (("23", A0), ("12", B0)):
        w = np.zeros(n); have = los[t] & need; thin = have & (Mb > 0)
        w[thin] = gap * (-D[t][thin]) / (Mb[thin] - D[t][thin])
        w[have & (Mb <= 0)] = WIDE
        slope = np.median((Mb[thin] - D[t][thin]) / gap) if thin.any() else np.nan
        rest = los[t] & ~need
        if slope_rule == "tangent":
            slope = np.median(tan[t][1]); thin2 = los[t] & ~(have & (Mb <= 0))
            w[thin2] = -D[t][thin2] / slope
            if not n_tan:   # own tangent where it is positive
                li, sl = tan[t]; ok = sl > 0; own = np.zeros(n, bool); own[li[ok]] = True; own &= thin2
                sl_full = np.zeros(n); sl_full[li] = sl; w[own] = -D[t][own] / sl_full[own]
        else:
            w[rest] = -D[t][rest] / slope
        widths[t] = w
    tt = lambda x: torch.tensor(x, device=c.DEV, dtype=c.dt)
    acc = bl.accuracy(tt(widths["23"]), tt(widths["12"]), law)
    runs = len(X23) + len(X12) + len(Xb) + sum(len(x) for x in Xt)
    cost = dict(runs=runs, steps=step_cost([X23, X12, Xb]) + sum(len(x) for x in Xt))   # tangents counted as full runs
    if return_parts: return acc, cost, dict(D=D, Mb=Mb, widths=widths, need=need)
    return acc, cost
