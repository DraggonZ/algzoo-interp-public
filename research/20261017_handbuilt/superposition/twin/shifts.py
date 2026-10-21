"""Nudge faithfulness as lead shifts. For each nudge test, on its lists, the change in the lead of the earlier contender
over the later one (or of the step-9 contender, or of the largest over the second), divided by the second's own vote
(its score coefficient times its value, from the exact per-list linear map). Median over lists, ARC beside the twins.
Rates (battery.py) mix this shift with how blurred a network's near ties are; shifts compare the mechanisms directly."""
import sys; sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family"); sys.path.insert(0, "../v23")
import numpy as np, torch
from common import DEV, dt, jac, top3, T, ARC
from battery import NT, NL, NS, N3, RL, ARC_GROUPS, hb_groups
def scores_n(W, X, nudges=()):
    h = torch.zeros(len(X), W[0].shape[0], device=DEV, dtype=dt)
    for t in range(10):
        h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
        for g, st, f in nudges:
            if t in st: h = h.clone(); h[:, g] = h[:, g] * f
    return h @ W[2].T
def unit(W, X, s):
    n = torch.arange(len(X), device=DEV); J = jac(W, X); return (J[n, s, s] * X[n, s]).abs().clamp(min=1e-9)
def shift(W, X, a, b, s, nudges):
    n = torch.arange(len(X), device=DEV); S0 = scores_n(W, X); S1 = scores_n(W, X, nudges)
    d = ((S1[n, a] - S1[n, b]) - (S0[n, a] - S0[n, b])) / unit(W, X, s)
    return d.median().item()
def shifts(W, G):
    r = {}
    for T_ in (5,):
        for who, a, b in (("2nd early", NS, N3), ("3rd early", N3, NS)):
            m = (NL <= T_ - 3) & (a >= NL + 2) & (a <= T_ - 2) & (b >= T_ + 1); X = NT[m][:50000]; am, bm, sm = a[m][:50000], b[m][:50000], NS[m][:50000]
            for f in (0.95, 1.05): r[f"memory x{f} step 5, earlier minus later ({who})"] = shift(W, X, am, bm, sm, [(G["memory"], [T_], f)])
    for who, a, b in (("2nd early", NS, N3), ("3rd early", N3, NS)):
        m = (NL <= 3) & (a > NL) & (a <= 3) & (b >= 6); X = NT[m][:50000]; am, bm, sm = a[m][:50000], b[m][:50000], NS[m][:50000]
        r[f"store x1.01 step 5, earlier minus later ({who})"] = shift(W, X, am, bm, sm, [(G["store"], [5], 1.01)])
    for who, a in (("2nd", NS), ("3rd", N3)):
        m = a == 9; X = NT[m][:50000]; other = torch.where(NS[m] == 9, N3[m], NS[m])[:50000]; sm = NS[m][:50000]
        for f in (0.95, 1.05): r[f"keep x{f} all steps, step-9 contender minus other ({who} at 9)"] = shift(W, X, torch.full_like(other, 9), other, sm, [(G["keep"], range(10), f)])
    L, S, _ = top3(RL); m = L <= 3; X = RL[m][:50000]; lm, sm = L[m][:50000], S[m][:50000]
    for f in (0.9, 0.8): r[f"hiding x{f} step 5, largest minus second"] = shift(W, X, lm, sm, sm, [(G["hiding"], [5], f)])
    return r
if __name__ == "__main__":
    from algzoo.handbuilt_v18 import build_v18
    from algzoo.handbuilt_v19 import build_v19
    from algzoo.handbuilt_v23 import build_v23
    from algzoo.handbuilt_v23_from_v19 import build_v23f
    nets = {"ARC": (ARC, ARC_GROUPS)}
    for name, net in (("v18", build_v18()), ("v19", build_v19(kappa=0.5)), ("v23f", build_v23f()), ("v23", build_v23()),
                      ("v23 fade.99", build_v23(U_mem=1.7, rho=0.99, level_exact=False))):
        nets[name] = (T(net.params), hb_groups(net))
    res = {k: shifts(W, G) for k, (W, G) in nets.items()}
    print(f"{'lead shift, in units of the second’s vote (median)':72s}" + "".join(f"{k:>12s}" for k in res))
    for t in res["ARC"]: print(f"{t:72s}" + "".join(f"{res[k][t]:+12.3f}" for k in res))
    for k in list(res)[1:]:
        a = np.array([res["ARC"][t] for t in res["ARC"]]); b = np.array([res[k][t] for t in res["ARC"]])
        print(f"{k}: same sign as ARC in {np.mean(np.sign(a) == np.sign(b)):.0%} of tests; median ratio of sizes {np.median(np.abs(b) / np.abs(a)):.2f}")
