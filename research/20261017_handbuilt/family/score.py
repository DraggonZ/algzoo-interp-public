"""Shared scorer: grid accuracy and ARC's mistakes reproduced (../hilton/topdown grid, 40 x 48 x 48, no sampling),
plus near-tie behaviour on random lists (third within 5% of second, second > 0): share where the third wins, overall,
by x_L / x_S and by arrival order. Family reference: mean of five trained networks at 95-96%."""
import sys, numpy as np, torch
import os; _H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hilton"); sys.path.insert(0, os.path.join(_H, "topdown")); sys.path.insert(0, os.path.join(_H, "propagate"))
import store_ladder as sl
from common import *
FIVE = (103, 111, 115, 209, 210)
def grid(models):
    nV = len(sl.wtt); st = {}
    def add(k, v): st[k] = st.get(k, 0.0) + v
    for i in range(0, 720, 16):
        tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
        bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
        X = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
        for k in range(3): X[ar, pos[:, k]] = val[:, k]
        w = sl.wtt.repeat(len(tt)); y = pos[:, 1]; a0 = sl.run(X).argmax(1); wr = a0 != y
        add("w", (w * wr).sum().item()); add("all", w.sum().item()); add("wL", (w * wr * (a0 == pos[:, 0])).sum().item()); add("wT", (w * wr * (a0 == pos[:, 2])).sum().item())
        for nm, Ws in models.items():
            for W in Ws:
                a1 = (W(X) if callable(W) else scores(W, X)).argmax(1); c = 1 / len(Ws)
                add((nm, "acc"), c * (w * (a1 == y)).sum().item()); add((nm, "rep"), c * (w * wr * (a1 == a0)).sum().item())
                add((nm, "repL"), c * (w * wr * (a0 == pos[:, 0]) * (a1 == a0)).sum().item()); add((nm, "repT"), c * (w * wr * (a0 == pos[:, 2]) * (a1 == a0)).sum().item())
    return {nm: (100 * st[(nm, "acc")] / st["all"], 100 * st[(nm, "rep")] / st["w"], 100 * st[(nm, "repL")] / st["wL"], 100 * st[(nm, "repT")] / st["wT"]) for nm in models}
RB = [1.1, 1.25, 1.5, 2.0, 3.0]; RBN = ["<1.1", "1.1-1.25", "1.25-1.5", "1.5-2", "2-3", ">3"]; ORD = ["LST", "LTS", "SLT", "TLS", "STL", "TSL"]
def near(models, n=300000):
    X = near_lists(n, seed=2); L, S, Tq = top3(X); ar = torch.arange(len(X), device=DEV)
    rb = np.digitize((X[ar, L] / X[ar, S]).cpu().numpy(), RB)
    ords = np.array(["".join("LST"[j] for j in np.argsort(p)) for p in torch.stack([L, S, Tq], 1).cpu().numpy()])
    out = {}
    for nm, Ws in models.items():
        wv = np.mean([((W(X) if callable(W) else scores(W, X)).argmax(1) == Tq).double().cpu().numpy() for W in Ws], 0)
        out[nm] = [100 * wv.mean()] + [100 * wv[rb == b].mean() for b in range(6)] + [100 * wv[ords == o].mean() for o in ORD]
    return out
def report(models, family=True):
    if family: models = {**models, "5 trained (family)": [trained(s) for s in FIVE]}
    g = grid(models); nr = near(models)
    print(f"{'network':34s} {'acc':>6s} {'ARC mistakes: all':>17s} {'L':>5s} {'T':>5s} | near ties, third wins: all | by x_L/x_S " + " ".join(RBN) + " | by order " + " ".join(ORD))
    for nm in models:
        a, r, rl, rt = g[nm]; v = nr[nm]
        print(f"{nm:34s} {a:5.2f}% {r:16.1f}% {rl:4.1f}% {rt:4.1f}% | {v[0]:5.1f}% | " + " ".join(f"{x:4.0f}" for x in v[1:7]) + " | " + " ".join(f"{x:4.0f}" for x in v[7:]))
    return g, nr
