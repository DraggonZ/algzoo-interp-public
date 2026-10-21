"""r13b look 5: does the zero-sum offset change which next step the push should use?
Push variants: h_11 with x_new = 0 (r12a), h_11 with a fresh x_new, the shifted list y = (x_1..x_9, new) with new = 0
or fresh. Each with no offset, d = -row_0/9, and (diagnostic, reads row 9) d = (g Push(row_9) - row_0)/9.
Rows 1-8 = g Push(row_{k-1}) + d, priced by the explorer on r12a's covered set, gain refitted."""
import numpy as np
import torch as th
import r12a_x as X

R = X.load("rules/r12a_regpush.py")
WI, WH, WO = X.WI, X.WH, X.WO
names = list(R.COVERED)
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
nexts = {
    "h11, new 0": np.maximum(h10 @ WH.T, 0),
    "h11, new fresh": np.maximum(h10 @ WH.T + x[:, 10, None] * WI[None, :], 0),
    "shifted, new 0": run(np.concatenate([x[:, 1:10], np.zeros((N, 1))], 1)),
    "shifted, new fresh": run(x[:, 1:11]),
}


class Var:
    def __init__(self, M, dmode):
        self.COVERED = R.COVERED; self.ORDER = R.ORDER; self.KINDS = R.KINDS
        self.STRUCTURE_BITS = R.STRUCTURE_BITS; self.init_constants = R.init_constants
        self.M = th.tensor(M); self.dmode = dmode
        self.idx = {k: th.tensor([names.index(f"{i}->out{k}") for i in range(16)]) for k in range(1, 9)}
        self.WO = th.tensor(WO)

    def weights(self, C):
        w = R.weights(C).clone()
        g = C[0]
        if self.dmode == "none":
            d = th.zeros(16, dtype=th.float64)
        elif self.dmode == "row0/9":
            d = -self.WO[0] / 9
        else:
            d = (g * self.M @ self.WO[9] - self.WO[0]) / 9
        for k in range(1, 9):
            w[self.idx[k]] = g * self.M @ self.WO[k - 1] + d
        return w


for lab, hn in nexts.items():
    M = np.linalg.solve(G, h10.T @ hn / N)
    for dm in ("none", "row0/9", "full [reads row 9]"):
        net, dd = X.price(Var(M, dm))
        errs = " ".join(f"{e / s:.2f}" for r, (e, s) in sorted(dd["rowerr"].items()) if r.startswith("out"))
        print(f"{lab:20s} {dm:20s} net {net:+.1f} gain {dd['C'][0]:.3f} | rows 1-9 rel err {errs}")
