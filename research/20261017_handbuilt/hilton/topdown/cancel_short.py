"""A short rule for the cancelling neurons (1, 6, 7, 8) from their exact linear map. On each list the input
(pre-activation) of each neuron at step t is exactly sum_s Jpre(t, s) x_s. Group the weights by the age a of the
largest L (steps since it arrived) and by each number's role: L, the previous largest P, the k-th number since L
(k = 1..a, the last one being the arriving number), and earlier numbers by how many steps before L (j = 1..). Rule:
neuron = ReLU(sum over roles of the median weight x the number). The ReLU gives the partial switch-offs. Scored on
the grid by mistakes reproduced (all, coin flips, middle, clear), cancelling neurons replaced, rest real."""
import numpy as np, torch
import store_ladder as sl
DEV = "cuda"; dt = torch.float64
wi, wh, wo = sl.W; C = [1, 6, 7, 8]
N = 200000; X = torch.randn(N, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(43))
h = torch.zeros(N, 16, device=DEV, dtype=dt); J = torch.zeros(N, 10, 16, device=DEV, dtype=dt); JP = []
for t in range(10):
    Jpre = J @ wh.T; Jpre[:, t, :] += wi; JP.append(Jpre[:, :, C].clone())
    a = h @ wh.T + X[:, t:t + 1] * wi; on = (a > 0).to(dt); J = Jpre * on[:, None, :]; h = a * on
JP = torch.stack(JP, 1)                                    # [N, t, s, 4]
def roles(X):
    n = len(X); R = torch.full((n, 10), -1, device=DEV, dtype=torch.long); RP = R.clone()
    r = torch.full((n,), -1, device=DEV, dtype=torch.long); rp = r.clone(); m = torch.zeros(n, device=DEV, dtype=dt)
    for t in range(10):
        R[:, t] = r; RP[:, t] = rp; new = X[:, t] > m; rp = torch.where(new, r, rp); m = torch.where(new, X[:, t], m); r = torch.where(new, t, r)
    return R, RP
R, RP = roles(X); ar = torch.arange(N, device=DEV)
# table: key (t, a) -> weights per role: role ids: 0 = L, 1 = P, 2.. = position offsets for others (s - r), stored as dict
TAB = {}
for t in range(1, 10):
    for a in range(1, t + 1):
        idx = torch.where(R[:, t] == t - a)[0]
        if len(idx) < 200: continue
        r = t - a; W = {}
        W["L"] = JP[idx, t, r].median(0).values
        hp = RP[idx, t] >= 0
        W["P"] = JP[idx[hp], t, RP[idx[hp], t]].median(0).values if hp.sum() > 50 else torch.zeros(4, device=DEV, dtype=dt)
        for s in range(10):
            if s == r or s > t: continue
            notP = RP[idx, t] != s
            W[s] = JP[idx[notP], t, s].median(0).values if notP.sum() > 50 else torch.zeros(4, device=DEV, dtype=dt)
        TAB[(t, a)] = W
nnum = sum(len(W) * 4 for W in TAB.values()); print(f"rule size: {nnum} numbers ({len(TAB)} situations)")


def pre_rule(Xg, t, R_, RP_):
    n = len(Xg); out = torch.zeros(n, 4, device=DEV, dtype=dt); arn = torch.arange(n, device=DEV)
    a = torch.where(R_[:, t] >= 0, t - R_[:, t], 0)
    for (tt_, aa), W in TAB.items():
        if tt_ != t: continue
        idx = torch.where(a == aa)[0]
        if len(idx) == 0: continue
        r = t - aa; v = Xg[idx, r, None] * W["L"]
        hasP = RP_[idx, t] >= 0; Pv = torch.where(hasP, Xg[idx, RP_[idx, t].clamp(min=0)], 0.0)
        v = v + Pv[:, None] * W["P"]
        for s in range(t + 1):
            if s == r: continue
            isP = RP_[idx, t] == s
            v = v + torch.where(isP, 0.0, Xg[idx, s])[:, None] * W[s]
        out[idx] = v
    noL = a == 0
    if noL.any(): out[noL] = Xg[noL, t, None] * wi[C]                  # before any record: input weight only (no positive number yet)
    return out


def run_rule(Xg):
    n = len(Xg); h = torch.zeros(n, 16, device=DEV, dtype=dt); R_, RP_ = roles(Xg)
    for t in range(10):
        h = torch.relu(h @ wh.T + Xg[:, t:t + 1] * wi)
        if t >= 1: h[:, C] = torch.relu(pre_rule(Xg, t, R_, RP_))
    return h @ wo.T


nV = len(sl.wtt); rec = []; M, Wt = [], []
for i in range(0, 720, 16):
    tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
    bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
    Xg = bmu[:, None].repeat(1, 10); arn = torch.arange(n, device=DEV)
    for k in range(3): Xg[arn, pos[:, k]] = val[:, k]
    w = sl.wtt.repeat(len(tt)); y = pos[:, 1]; S0 = sl.run(Xg); a0 = S0.argmax(1); wr = a0 != y
    a1 = run_rule(Xg).argmax(1)
    M.append((S0[arn, a0] - S0[arn, y])[wr].cpu().numpy()); Wt.append(w[wr].cpu().numpy()); rec.append(((a1 == a0)[wr]).cpu().numpy())
    if i == 0: agr = acc = Z = 0.0
    agr += (w * (a1 == a0)).sum().item(); acc += (w * (a1 == y)).sum().item(); Z += w.sum().item()
M, Wt, r = np.concatenate(M), np.concatenate(Wt), np.concatenate(rec); band = np.digitize(M, [0.89, 2.41])
print(f"cancelling neurons by median exact weights: mistakes reproduced all {100*(Wt*r).sum()/Wt.sum():.1f}%, coin flips {100*(Wt[band==0]*r[band==0]).sum()/Wt[band==0].sum():.1f}%, middle {100*(Wt[band==1]*r[band==1]).sum()/Wt[band==1].sum():.1f}%, clear {100*(Wt[band==2]*r[band==2]).sum()/Wt[band==2].sum():.1f}%; agreement {100*agr/Z:.2f}%, accuracy {100*acc/Z:.2f}%")
