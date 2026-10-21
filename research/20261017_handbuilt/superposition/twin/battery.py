"""Faithfulness by nudge tests: ARC's steering tests (steer/RESULTS.md) as one fixed battery, run on any 16-neuron
network given which neurons form its store (largest so far: store and detector), keep gate (ARC's neuron 5), hiding
group and memory. Each test reports a rate; the twin is faithful where its rates match ARC's. Lists and seeds are fixed."""
import sys; sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family")
import numpy as np, torch
from common import DEV, dt, T, ARC, top3, near_lists
ARC_GROUPS = {"store": [2, 4], "keep": [5], "hiding": [1, 6, 7, 8], "memory": [0, 3, 10, 11, 12, 13, 14, 15]}
def hb_groups(net):
    I = net.idx; mem = [I[k] for k in I if k.startswith("mem")]
    hid = [I[k] for k in I if k.startswith("clock") or k == "flip"]
    return {"store": [I["store"], I["detect"]], "keep": [I["keep"]], "hiding": hid, "memory": mem}
def run(W, X, nudges=(), extra=None):
    """nudges: (neurons, steps, factor); extra: inputs appended after the ten (tensor [n, k])."""
    Z = X if extra is None else torch.cat([X, extra], 1)
    h = torch.zeros(len(X), W[0].shape[0], device=DEV, dtype=dt)
    for t in range(Z.shape[1]):
        h = torch.relu(h @ W[1].T + Z[:, t:t + 1] * W[0])
        for g, st, f in nudges:
            if t in st: h = h.clone(); h[:, g] = h[:, g] * f
    return (h @ W[2].T).argmax(1)
def rnd(n, seed):
    g = torch.Generator(device=DEV).manual_seed(seed); return torch.randn(n, 10, device=DEV, dtype=dt, generator=g)
NT = near_lists(2_000_000, seed=3); NL, NS, N3 = top3(NT)
RL = rnd(500_000, 11); RLl, RLs, RL3 = top3(RL)
def pc(m): return 100 * m.double().mean().item()
def battery(W, G):
    r = {}
    # memory and store nudges on near ties: largest first, earlier contender in memory, later after step t
    for T_ in (5, 7):
        for who, a, b in (("2nd early", NS, N3), ("3rd early", N3, NS)):
            m = (NL <= T_ - 3) & (a >= NL + 2) & (a <= T_ - 2) & (b >= T_ + 1); X, am = NT[m], a[m]
            r[f"near tie, step {T_}, {who}: earlier wins"] = pc(run(W, X) == am)
            for f in (0.95, 1.05): r[f"  memory x{f} at step {T_} ({who})"] = pc(run(W, X, [(G["memory"], [T_], f)]) == am)
    for who, a, b in (("2nd early", NS, N3), ("3rd early", N3, NS)):
        m = (NL <= 3) & (a > NL) & (a <= 3) & (b >= 6); X, am = NT[m], a[m]
        r[f"near tie, largest and {who} by step 3, other after 5: earlier wins"] = pc(run(W, X) == am)
        r[f"  store x1.01 at step 5 ({who})"] = pc(run(W, X, [(G["store"], [5], 1.01)]) == am)
        for f in (0.9, 1.1):
            Xc = X.clone(); Xc[torch.arange(len(X), device=DEV), top3(X)[0]] *= f
            r[f"  largest's value x{f} ({who})"] = pc(run(W, Xc) == am)
    # keep gate at every step: last-step contenders
    for who, a in (("2nd", NS), ("3rd", N3)):
        m = a == 9; X = NT[m]
        r[f"near tie, {who} at step 9: it wins"] = pc(run(W, X) == 9)
        for f in (0.95, 1.05): r[f"  keep gate x{f}, all steps ({who} at 9)"] = pc(run(W, X, [(G["keep"], range(10), f)]) == 9)
    X = NT[:300_000]; a0 = run(W, X)
    r["near ties: answers changed by keep gate x1.05, all steps"] = pc(run(W, X, [(G["keep"], range(10), 1.05)]) != a0)
    # late bias and its removal by a smaller last input
    m = (N3 == 9) | (NS == 9); X = NT[m]; s9 = NS[m] == 9
    a = run(W, X); r["late bias: step-9 contender wins (truly second half the time)"] = pc(a == 9)
    Xc = X.clone(); Xc[:, 9] *= 0.99; r["  last input x0.99"] = pc(run(W, Xc) == 9)
    # hiding
    m = RLl <= 3; X, lm = RL[m], RLl[m]
    r["largest by step 3: largest named"] = pc(run(W, X) == lm)
    for f in (0.9, 0.8): r[f"  hiding x{f} at step 5"] = pc(run(W, X, [(G["hiding"], [5], f)]) == lm)
    r["  hiding silenced at all steps (all lists)"] = pc(run(W, RL, [(G["hiding"], range(10), 0.0)]) == RLl)
    # extra and appended numbers
    ok = RLs >= 1; z = torch.zeros(len(RL), 1, device=DEV, dtype=dt); mx = RL.max(1).values[:, None]; v = RL.sort(1, descending=True).values
    r["extra input 0: names second-1"] = pc((run(W, RL, extra=z) == RLs - 1)[ok])
    r["append 2 x largest: names largest-1"] = pc((run(W, RL, extra=2 * mx) == RLl - 1)[RLl >= 1])
    r["append between largest and second: names 9"] = pc(run(W, RL, extra=(v[:, :1] + v[:, 1:2]) / 2) == 9)
    r["append -3: names second-1"] = pc((run(W, RL, extra=z - 3) == RLs - 1)[ok])
    # sensitivity map: answers changed, x1.02 one group at one step
    a0 = run(W, RL[:200_000]); sm = []
    for g in ("store", "keep", "hiding", "memory"):
        for t in range(10): sm.append(pc(run(W, RL[:200_000], [(G[g], [t], 1.02)]) != a0))
    r["_sensitivity"] = np.array(sm)
    r["accuracy"] = pc(run(W, RL) == RLs)
    return r
def compare(results, ref="ARC"):
    names = [k for k in results[ref] if not k.startswith("_")]
    print(f"{'test':72s}" + "".join(f"{n:>12s}" for n in results))
    for k in names: print(f"{k:72s}" + "".join(f"{results[n][k]:12.1f}" for n in results))
    A = results[ref]
    for n, R in results.items():
        if n == ref: continue
        d = np.array([abs(R[k] - A[k]) for k in names if k != "accuracy"])
        sa, sb = np.log10(A["_sensitivity"] + 0.01), np.log10(R["_sensitivity"] + 0.01)
        print(f"{n}: mean |rate - ARC's| {d.mean():.1f} points over {len(d)} tests; within 5 points {np.mean(d <= 5):.0%}; "
              f"sensitivity map (40 cells): correlation of logs {np.corrcoef(sa, sb)[0, 1]:.2f}, median ratio {np.median((R['_sensitivity'] + 0.01) / (A['_sensitivity'] + 0.01)):.2f}")
if __name__ == "__main__":
    from algzoo.handbuilt_v18 import build_v18
    from algzoo.handbuilt_v19 import build_v19
    from algzoo.handbuilt_v23 import build_v23
    nets = {"ARC": (ARC, ARC_GROUPS)}
    for name, net in (("v18", build_v18()), ("v19", build_v19(kappa=0.5)), ("v23", build_v23())):
        nets[name] = (T(net.params), hb_groups(net))
    res = {n: battery(W, G) for n, (W, G) in nets.items()}
    compare(res)
    np.save("battery_base.npy", res, allow_pickle=True)
