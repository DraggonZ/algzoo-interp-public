"""Tie-band estimator with band widths from a safeguarded Newton search along each gap (deterministic).

Nodes as in cheap_est.py (u0 = 0.78, a0 = 0.37, b0 = 0.58, background -0.40). For each tie configuration one run
gives the margin D of both orderings sharing it. For each ordering wrong at the tie (D < 0):
  1. a forward tangent at the tie along the gap gives the slope s; the zero of the margin on the tie's own linear
     piece is w1 = -D / s;
  2. if s > 0 and w1 <= ACCEPT, the width is w1 (no more runs);
  3. otherwise search: evaluate the margin (and its tangent) at a trial gap (w1 capped at AMAX, or STEP if s <= 0);
     if the margin is still negative, move on by Newton (or double the gap if the slope is <= 0), up to MAXIT
     evaluations; once a trial gap gives a right answer, the width is the secant zero between the last wrong and
     the first right gap; if none does, the band is wide (width WIDE, the whole gap range of the integration grid).
Cost: tie runs + one tangent per wrong ordering + two runs (forward and tangent) per search evaluation. The band
model and the integral are bandlib.accuracy (scale law 'norm', second-largest below 0 counted wrong)."""
import numpy as np, torch
import common as c, bandlib as bl, cheap_est as ce

ACCEPT, AMAX, STEP, MAXIT, WIDE = 0.1, 5.0, 0.2, 4, 5.0


def margins_at(W, P, nd, tie, g):
    """Margin and its slope along the gap, for triples P at gap g [n] from the tie node."""
    X = ce._X(P, nd); n = len(P); ar = torch.arange(n, device=c.DEV); d = torch.zeros_like(X)
    d[ar, P[:, 0]] = 1
    if tie == "23": d[ar, P[:, 1]] = 1
    X = X + g[:, None] * d
    S, dS = c.scores(W, X, d); m, sl, _ = c.margin_from(S, P[:, 1], dS)
    return m, sl


def estimate(W, accept=ACCEPT, law="norm", return_parts=False):
    T = c.T; Tn = T.cpu().numpy(); n = 720
    keys = {"23": sorted({(p1, min(p2, p3), max(p2, p3)) for p1, p2, p3 in Tn.tolist()}),
            "12": sorted({(min(p1, p2), max(p1, p2), p3) for p1, p2, p3 in Tn.tolist()})}
    nodes = {"23": ce.N23, "12": ce.N12}
    runs = 0; widths = {}; Xall = []; tan_steps = 0.0; info = {}
    for tie in ("23", "12"):
        P = torch.tensor(keys[tie], device=c.DEV); X = ce._X(P, nodes[tie]); Xall.append(X); runs += len(X)
        S, _ = c.scores(W, X); Sn = S.cpu().numpy(); kidx = {k: i for i, k in enumerate(keys[tie])}
        D = np.zeros(n)
        for i, (p1, p2, p3) in enumerate(Tn.tolist()):
            k = (p1, min(p2, p3), max(p2, p3)) if tie == "23" else (min(p1, p2), max(p1, p2), p3)
            s = Sn[kidx[k]]; D[i] = s[p2] - np.delete(s, p2).max()
        li = np.where(D < 0)[0]; Pl = T[torch.tensor(li, device=c.DEV)]
        g0 = torch.zeros(len(li), device=c.DEV, dtype=c.dt)
        m0, s0 = margins_at(W, Pl, nodes[tie], tie, g0); runs += len(li)
        first = Pl[:, 0] if tie == "12" else torch.minimum(Pl[:, 0], Pl[:, 1])
        tan_steps += (10 - first).sum().item() / 10
        m0 = m0.cpu().numpy(); s0 = s0.cpu().numpy()
        w = np.zeros(n); ok = (s0 > 0) & (-m0 / np.where(s0 > 0, s0, 1) <= accept)
        w[li[ok]] = -m0[ok] / s0[ok]
        todo = np.where(~ok)[0]                     # indices into li
        a_lo = np.zeros(len(todo)); m_lo = m0[todo]; s_lo = s0[todo]; done = np.zeros(len(todo), bool); wt = np.full(len(todo), WIDE)
        for it in range(MAXIT):
            act = np.where(~done)[0]
            if len(act) == 0: break
            trial = np.where(s_lo[act] > 0, a_lo[act] - m_lo[act] / np.where(s_lo[act] > 0, s_lo[act], 1), np.maximum(2 * a_lo[act], STEP))
            trial = np.minimum(np.maximum(trial, a_lo[act] + 1e-3), AMAX)
            mt, st = margins_at(W, Pl[torch.tensor(todo[act], device=c.DEV)], nodes[tie], tie, torch.tensor(trial, device=c.DEV, dtype=c.dt))
            runs += 2 * len(act); mt = mt.cpu().numpy(); st = st.cpu().numpy()
            right = mt > 0
            j = act[right]; wt[j] = a_lo[j] + (trial[right] - a_lo[j]) * (-m_lo[j]) / (mt[right] - m_lo[j]); done[j] = True
            j = act[~right]; a_lo[j] = trial[~right]; m_lo[j] = mt[~right]; s_lo[j] = st[~right]
            done[act[(~right) & (trial >= AMAX)]] = True
        w[li[todo]] = wt
        widths[tie] = w; info[tie] = dict(n_losers=len(li), n_search=len(todo))
    tt = lambda x: torch.tensor(x, device=c.DEV, dtype=c.dt)
    acc = bl.accuracy(tt(widths["23"]), tt(widths["12"]), law)
    steps = ce.step_cost(Xall) + tan_steps + (runs - sum(len(x) for x in Xall) - sum(v["n_losers"] for v in info.values()))
    cost = dict(runs=runs, steps=steps, **{f"search{t}": info[t]["n_search"] for t in info})
    return (acc, cost, widths) if return_parts else (acc, cost)
