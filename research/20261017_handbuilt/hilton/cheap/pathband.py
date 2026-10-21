"""Exact band widths at the tie nodes by following the network's linear pieces along the gap (deterministic).

The network has no biases and ReLUs only, so along a straight line in input space every pre-activation and every
score is piecewise linear in the distance a moved. Starting at a tie (a = 0, where the ordering's margin D < 0), one
tangent pass gives, for the current on/off pattern, the slope of all 160 pre-activations and all 10 scores. The next
pattern change is at the smallest a where some pre-activation reaches 0; the margin (true second's score minus the
best other) reaches 0 inside the piece where every score below it rises past it. If that comes first, it is the band
width, exactly; otherwise move to the pattern change, flip, and take the next tangent pass. Pre-activation values at
the pattern change follow from the linear piece, so each piece costs one tangent pass (counted as one run; in RNN
steps, only the steps from the changed neuron's step onward).

Nodes, tie runs and the band model are those of cheap_est.py / bandlib.py. Width cap WIDE (5) after MAXP pieces or
when a passes AMAX."""
import numpy as np, torch
import common as c, bandlib as bl, cheap_est as ce

AMAX, MAXP, WIDE = 5.0, 60, 5.0


def _pass(W, X, d):
    """Forward values and tangents of all pre-activations [n,10,16] and scores [n,10]."""
    wi, wh, wo = W; n = X.shape[0]
    h = torch.zeros(n, wi.shape[0], device=c.DEV, dtype=c.dt); dh = torch.zeros_like(h); Z = []; dZ = []
    for t in range(10):
        z = h @ wh.T + X[:, t:t + 1] * wi; dz = dh @ wh.T + d[:, t:t + 1] * wi
        Z.append(z); dZ.append(dz); on = z > 0
        h = torch.where(on, z, 0.0); dh = torch.where(on, dz, 0.0)
    return torch.stack(Z, 1), torch.stack(dZ, 1), h @ wo.T, dh @ wo.T


def widths(W, tie, P, nd, eps=1e-9, maxp=MAXP):
    """P [n,3] triples wrong at their tie. Returns widths [n], pieces used [n], tangent-pass steps [n]."""
    n = len(P); ar = torch.arange(n, device=c.DEV)
    X0 = ce._X(P, nd); d = torch.zeros_like(X0); d[ar, P[:, 0]] = 1
    if tie == "23": d[ar, P[:, 1]] = 1
    a = torch.zeros(n, device=c.DEV, dtype=c.dt); w = torch.full((n,), WIDE, device=c.DEV, dtype=c.dt)
    pieces = torch.zeros(n, device=c.DEV); steps = torch.zeros(n, device=c.DEV, dtype=c.dt)
    kink_step = torch.minimum(P[:, 0], P[:, 1] if tie == "23" else P[:, 0]).to(c.dt)   # first step the tangent is nonzero
    active = torch.ones(n, dtype=torch.bool, device=c.DEV)
    for it in range(maxp):
        idx = active.nonzero().squeeze(1)
        if len(idx) == 0: break
        aa = a[idx]
        Z, dZ, S, dS = _pass(W, X0[idx] + aa[:, None] * d[idx], d[idx])
        pieces[idx] += 1; steps[idx] += (10 - kink_step[idx]) / 10
        # next pattern change
        cross = torch.where(((Z > 0) & (dZ < 0)) | ((Z < 0) & (dZ > 0)), -Z / dZ, torch.full_like(Z, np.inf))
        cross = cross.clamp_min(0).reshape(len(idx), -1); nxt, arg = cross.min(1)
        # margin root inside this piece: every other score below the true second's must be passed
        p2 = P[idx, 1]; r = torch.arange(len(idx), device=c.DEV)
        m = S[r, p2][:, None] - S; dm = dS[r, p2][:, None] - dS; m[r, p2] = 1.0; dm[r, p2] = 0.0
        need = m <= 0
        ok = ~(need & (dm <= 0)).any(1)
        root = torch.where(need, -m / dm.where(dm > 0, torch.ones_like(dm)), torch.zeros_like(m)).max(1).values
        root = torch.where(ok, root, torch.full_like(root, np.inf))
        done = root <= nxt
        w[idx[done]] = aa[done] + root[done]
        far = (~done) & (aa + nxt >= AMAX)
        active[idx[done | far]] = False
        mv = ~(done | far)
        if it == maxp - 1 and maxp < MAXP:
            # out of pieces: extrapolate this piece's straight line to the margin's zero (wide if it never gets there)
            ext = torch.where(ok, aa + root, torch.full_like(root, WIDE)).clamp_max(WIDE)
            w[idx[mv]] = ext[mv]; break
        a[idx[mv]] = aa[mv] + nxt[mv] * (1 + eps) + eps
        kink_step[idx[mv]] = (arg[mv] // 16).to(c.dt)
    return w, pieces, steps


def estimate(W, law="norm", return_parts=False, maxp=MAXP):
    T = c.T; Tn = T.cpu().numpy(); n = 720
    keys = {"23": sorted({(p1, min(p2, p3), max(p2, p3)) for p1, p2, p3 in Tn.tolist()}),
            "12": sorted({(min(p1, p2), max(p1, p2), p3) for p1, p2, p3 in Tn.tolist()})}
    nodes = {"23": ce.N23, "12": ce.N12}; out = {}; runs = 0; steps = 0.0; Xs = []; npieces = 0
    for tie in ("23", "12"):
        P = torch.tensor(keys[tie], device=c.DEV); X = ce._X(P, nodes[tie]); Xs.append(X); runs += len(X)
        S, _ = c.scores(W, X); Sn = S.cpu().numpy(); kidx = {k: i for i, k in enumerate(keys[tie])}
        D = np.zeros(n)
        for i, (p1, p2, p3) in enumerate(Tn.tolist()):
            k = (p1, min(p2, p3), max(p2, p3)) if tie == "23" else (min(p1, p2), max(p1, p2), p3)
            s = Sn[kidx[k]]; D[i] = s[p2] - np.delete(s, p2).max()
        li = np.where(D < 0)[0]
        wl, pc, st = widths(W, tie, T[torch.tensor(li, device=c.DEV)], nodes[tie], maxp=maxp)
        w = np.zeros(n); w[li] = wl.cpu().numpy(); out[tie] = w
        runs += int(pc.sum().item()); steps += st.sum().item(); npieces += int(pc.sum().item())
    tt = lambda x: torch.tensor(x, device=c.DEV, dtype=c.dt)
    acc = bl.accuracy(tt(out["23"]), tt(out["12"]), law)
    cost = dict(runs=runs, steps=ce.step_cost(Xs) + steps, pieces=npieces)
    return (acc, cost, out) if return_parts else (acc, cost)
