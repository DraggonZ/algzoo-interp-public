"""r12a look 7: the shifted-list push. Fair votes compare two 10-number lists: x = (x_0 .. x_9) read by row k, and
y = (x_1 .. x_9, new) read by row k-1, where x_k sits at position k-1. So row k = argmin E[(r . h_10(x) -
row_{k-1} . h_10(y))^2]. Unlike h_11 = ReLU(W h_10(x)), h_10(y) has no trace of x_0 (the number that left the list).
new = 0 or a fresh normal. Price on the current best, gain refitted, rows 1-8, all 16 columns."""
import numpy as np
import torch as th
import r12a_x as X

WI, WH, WO = X.WI, X.WH, X.WO
R = X.load(X.BEST)
net0, _ = X.price(R)
N = 400_000
rng = np.random.default_rng(101)
x = rng.standard_normal((N, 11))


def run(xs):
    h = np.zeros((len(xs), 16))
    for t in range(xs.shape[1]):
        h = np.maximum(h @ WH.T + xs[:, t, None] * WI[None, :], 0)
    return h


h10 = run(x[:, :10])
G = h10.T @ h10 / N
T = [0, 3, 10, 11, 12, 13, 14, 15, 1, 6, 7, 8, 2, 5]
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
for mode in ("zero", "fresh"):
    y = np.concatenate([x[:, 1:10], (x[:, 10:11] if mode == "fresh" else np.zeros((N, 1)))], 1)
    hy = run(y)
    REG = {k: np.linalg.solve(G, h10.T @ (hy @ WO[k - 1]) / N) for k in range(1, 10)}
    print(mode)
    for k in range(1, 10):
        r = REG[k]; e = WO[k] - r
        gb = (r @ WO[k]) / (r @ r)
        print(f"  row {k}: err all 16 {np.linalg.norm(e) / np.linalg.norm(WO[k]):.3f}, push cols {np.linalg.norm(e[T]) / np.linalg.norm(WO[k, T]):.3f}, "
              f"best scale {gb:.3f}, residual cos with row 0 {cos(e[T], WO[0, T]):+.2f}")
    ex = {}
    for k in range(1, 9):
        for i in range(16):
            ex[f"{i}->out{k}"] = (lambda C, bw, v=float(REG[k][i]): C[0] * v)
    net, d = X.price(X.Extend(R, ex))
    print(f"  price rows 1-8 all 16: {net - net0:+.1f} (gain {d['C'][0]:.3f})")
    for r_, (e_, s_) in sorted(d["rowerr"].items()):
        if r_.startswith("out"):
            print(f"     {r_} rel err {e_ / s_:.3f}")
