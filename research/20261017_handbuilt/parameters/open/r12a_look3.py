"""r12a look 3: price the regression push (look 2) on the current best, gain refitted.
Variants: rows 1-8 on the push columns (TGT) replaced by C * r_k (C = the base's gain constant, refitted)."""
import sys
import numpy as np
import torch as th
import r12a_x as X

WI, WH, WO = X.WI, X.WH, X.WO
R = X.load(X.BEST)
net0, _ = X.price(R)
N = 400_000
g = th.Generator().manual_seed(101)
x = th.randn(N, 11, generator=g, dtype=th.float64)
Wi, Wh = th.tensor(WI), th.tensor(WH)
h = th.zeros(N, 16, dtype=th.float64)
for t in range(10):
    h = th.relu(h @ Wh.T + x[:, t, None] * Wi)
h10 = h.numpy()
TGT = [0, 3, 10, 11, 12, 13, 14, 15, 1, 6, 7, 8, 2, 5]
G = h10.T @ h10 / N
REG = {}
for mode in ("fresh", "zero"):
    xin = x[:, 10, None] * Wi if mode == "fresh" else 0
    h11 = th.relu(h @ Wh.T + xin).numpy()
    for k in range(1, 10):
        REG[mode, k] = np.linalg.solve(G, h10.T @ (h11 @ WO[k - 1]) / N)


def cand(mode, rows, cols, scaled):
    ex = {}
    for k in rows:
        for i in cols:
            v = float(REG[mode, k][i])
            ex[f"{i}->out{k}"] = (lambda C, bw, v=v: C[0] * v) if scaled else v
    return ex


for mode in (("fresh", "zero") if __name__ == "__main__" else ()):
    for rows in (range(1, 9), [1, 3, 5, 6, 7, 8], [3, 6, 7]):
        for scaled in (True, False):
            net, d = X.price(X.Extend(R, cand(mode, rows, TGT, scaled)))
            print(f"{mode} rows {list(rows)} scaled {scaled}: {net - net0:+.1f} (gain {d['C'][0]:.3f})", flush=True)
