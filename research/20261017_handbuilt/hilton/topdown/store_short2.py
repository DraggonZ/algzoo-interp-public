"""Variant (store_short2): both weights by age and by P/L in five bins (0, 0.5, 0.7, 0.8, 0.9, 1). Based on: a short store rule from the mechanism found: store = cL(a) L + q(a, P > 0.5 L) P, a = steps since L arrived.
cL and q are medians of the store's exact per-number weights (store_jacobian.py method) by age and by whether the
previous largest P was above half the largest (27 numbers). Scored on the grid by mistakes reproduced, split into the
coin-flip, middle and clear thirds of mistakes (edges from mistakes_by_margin.py: 0.89, 2.41)."""
import numpy as np, torch
import store_ladder as sl, store_ladder2 as s2
DEV = "cuda"; dt = torch.float64
wi, wh, wo = sl.W
N = 200000; X = torch.randn(N, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(41))
h = torch.zeros(N, 16, device=DEV, dtype=dt); J = torch.zeros(N, 10, 16, device=DEV, dtype=dt); JS = []
for t in range(10):
    a = h @ wh.T + X[:, t:t + 1] * wi; on = (a > 0).to(dt); J = J @ wh.T; J[:, t, :] += wi; J = J * on[:, None, :]; h = a * on; JS.append(J[:, :, 2].clone())
JS = torch.stack(JS, 1)
def roles(X):
    n = len(X); R = torch.full((n, 10), -1, device=DEV, dtype=torch.long); RP = R.clone()
    r = torch.full((n,), -1, device=DEV, dtype=torch.long); rp = r.clone(); m = torch.zeros(n, device=DEV, dtype=dt)
    for t in range(10):
        R[:, t] = r; RP[:, t] = rp; new = X[:, t] > m; rp = torch.where(new, r, rp); m = torch.where(new, X[:, t], m); r = torch.where(new, t, r)
    return R, RP
R, RP = roles(X); ar = torch.arange(N, device=DEV)
cL = torch.zeros(10, device=DEV, dtype=dt); q = torch.zeros(10, 2, device=DEV, dtype=dt)
BINS = torch.tensor([0.5, 0.7, 0.8, 0.9], device=DEV, dtype=dt); cB = torch.zeros(10, 6, device=DEV, dtype=dt); qB = torch.zeros(10, 6, device=DEV, dtype=dt)
for a in range(1, 10):
    vals, pv = [], {0: [], 1: []}
    for t in range(a, 10):
        idx = torch.where((R[:, t] == t - a) & (X[ar, R[:, t].clamp(min=0)] > 0.3))[0]
        vals.append(JS[idx, t, t - a])
        hp = RP[idx, t] >= 0; idp = idx[hp]; ratio = X[idp, RP[idp, t]] / X[idp, t - a]
        for k in (0, 1):
            sel = idp[(ratio > 0.5) == bool(k)]
            if len(sel): pv[k].append(JS[sel, t, RP[sel, t]])
    cL[a] = torch.cat(vals).median()
    for t in range(a, 10):
        pass
    allidx = [(t, torch.where((R[:, t] == t - a) & (X[ar, R[:, t].clamp(min=0)] > 0.3))[0]) for t in range(a, 10)]
    cl_, ql_, kb_ = [], [], []
    for t, idx in allidx:
        hp = RP[idx, t] >= 0; ratio = torch.where(hp, X[idx, RP[idx, t].clamp(min=0)] / X[idx, t - a], torch.full_like(X[idx, 0], -1.0))
        kb = torch.where(hp, torch.bucketize(ratio, BINS) + 1, 0); cl_.append(JS[idx, t, t - a]); ql_.append(torch.where(hp, JS[idx, t, RP[idx, t].clamp(min=0)], 0.0)); kb_.append(kb)
    cl_, ql_, kb_ = torch.cat(cl_), torch.cat(ql_), torch.cat(kb_)
    for b in range(6):
        sel = kb_ == b
        if sel.sum() > 30: cB[a, b] = cl_[sel].median(); qB[a, b] = ql_[sel].median()
        else: cB[a, b] = cL[a]
    for k in (0, 1): q[a, k] = torch.cat(pv[k]).median() if pv[k] else 0.0
print("weight on the largest by age:", " ".join(f"{v:.1f}" for v in cL[1:9].tolist()))
print("weight on the previous largest, below half:", " ".join(f"{v:.2f}" for v in q[1:9, 0].tolist()), "| above half:", " ".join(f"{v:.2f}" for v in q[1:9, 1].tolist()))


def run_short(Xg, use_switch=True):
    n = len(Xg); h = torch.zeros(n, 16, device=DEV, dtype=dt); R, RP = roles(Xg); arn = torch.arange(n, device=DEV)
    for t in range(10):
        h = torch.relu(h @ wh.T + Xg[:, t:t + 1] * wi)
        a = torch.where(R[:, t] >= 0, t - R[:, t], 0).clamp(max=9)
        Lv = torch.where(R[:, t] >= 0, Xg[arn, R[:, t].clamp(min=0)], 0.0).clamp(min=0)
        Pv = torch.where(RP[:, t] >= 0, Xg[arn, RP[:, t].clamp(min=0)], 0.0).clamp(min=0)
        k = ((Pv > 0.5 * Lv) & use_switch).long() if use_switch else torch.zeros(n, device=DEV, dtype=torch.long)
        qq = q[a, k] if use_switch else (q[a, 0] + q[a, 1]) / 2
        h[:, 2] = torch.relu(cL[a] * Lv + qq * Pv)
    return h @ wo.T


def run_binned(Xg):
    n = len(Xg); h = torch.zeros(n, 16, device=DEV, dtype=dt); R, RP = roles(Xg); arn = torch.arange(n, device=DEV)
    for t in range(10):
        h = torch.relu(h @ wh.T + Xg[:, t:t + 1] * wi)
        a = torch.where(R[:, t] >= 0, t - R[:, t], 0).clamp(max=9)
        Lv = torch.where(R[:, t] >= 0, Xg[arn, R[:, t].clamp(min=0)], 0.0).clamp(min=0)
        Pv = torch.where(RP[:, t] >= 0, Xg[arn, RP[:, t].clamp(min=0)], 0.0).clamp(min=0)
        kb = torch.where(RP[:, t] >= 0, torch.bucketize(Pv / Lv.clamp(min=1e-9), BINS) + 1, 0)
        h[:, 2] = torch.relu(cB[a, kb] * Lv + qB[a, kb] * Pv)
    return h @ wo.T


models = [("both weights by age and by P/L bin (108 numbers)", run_binned), ("age table on the largest only (9 numbers)", lambda X: sl.run(X, rule=lambda L, a: sl.r[a.clamp(max=9)] * L)),
          ("largest + previous largest, no switch (18)", lambda X: run_short(X, False)),
          ("largest + previous largest with the half switch (27)", run_short),
          ("fitted correction (93)", s2.run4)]
nV = len(sl.wtt); rec = {k: [] for k, _ in models}; M, Wt = [], []
for i in range(0, 720, 16):
    tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
    bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
    Xg = bmu[:, None].repeat(1, 10); arn = torch.arange(n, device=DEV)
    for k in range(3): Xg[arn, pos[:, k]] = val[:, k]
    w = sl.wtt.repeat(len(tt)); y = pos[:, 1]; S0 = sl.run(Xg); a0 = S0.argmax(1); wr = a0 != y
    M.append((S0[arn, a0] - S0[arn, y])[wr].cpu().numpy()); Wt.append(w[wr].cpu().numpy())
    for k, f in models: rec[k].append((f(Xg).argmax(1) == a0)[wr].cpu().numpy())
M, Wt = np.concatenate(M), np.concatenate(Wt); band = np.digitize(M, [0.89, 2.41])
print(f"{'':54s} {'all':>7s} {'coin flips':>11s} {'middle':>8s} {'clear':>8s}")
for k, _ in models:
    r = np.concatenate(rec[k]); print(f"{k:54s} {100*(Wt*r).sum()/Wt.sum():6.1f}% " + " ".join(f"{100*(Wt[band==b]*r[band==b]).sum()/Wt[band==b].sum():9.1f}%" for b in range(3)))
