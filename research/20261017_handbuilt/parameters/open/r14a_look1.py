"""r14a look 1: variants of the push on top of r13b_gauge, priced by the explorer (gain refitted).
A: two-step push (row_k from row_{k-2} through two next steps) averaged with the one-step push.
B: target with a fresh arriving number, or averaged over fresh and zero."""
import numpy as np
import torch as th
import r12a_x as X

R = X.load("rules/r13b_gauge.py")
WI, WH, WO = X.WI, X.WH, X.WO
names = list(R.COVERED)
N = 400_000
rng = np.random.default_rng(101)
x = rng.standard_normal((N, 12))
h = np.zeros((N, 16))
for t in range(10):
    h = np.maximum(h @ WH.T + x[:, t, None] * WI[None, :], 0)
h10 = h
h11 = np.maximum(h10 @ WH.T, 0)
h12 = np.maximum(h11 @ WH.T, 0)
h11f = np.maximum(h10 @ WH.T + x[:, 10, None] * WI[None, :], 0)
G = h10.T @ h10 / N
M1 = np.linalg.solve(G, h10.T @ h11 / N)
M2 = np.linalg.solve(G, h10.T @ h12 / N)
M1f = np.linalg.solve(G, h10.T @ h11f / N)
print("base", X.price(R)[0])
idx = {k: th.tensor([names.index(f"{i}->out{k}") for i in range(16)]) for k in range(1, 10) if f"0->out{k}" in names}
print("covered readout rows", sorted(idx))
W = th.tensor(WO)


class Var:
    def __init__(self, f, extra=0):
        self.COVERED = R.COVERED; self.ORDER = R.ORDER; self.KINDS = R.KINDS
        self.STRUCTURE_BITS = R.STRUCTURE_BITS + extra; self.init_constants = R.init_constants
        self.f = f

    def weights(self, C):
        w = R.weights(C).clone()
        self.f(w, C[0])
        return w


def mk(M, two=None, a=0.5):
    M = th.tensor(M); M2t = th.tensor(two) if two is not None else None

    def f(w, g):
        d = -W[0] / 9
        for k in range(1, 9):
            one = g * M @ W[k - 1] + d
            if M2t is not None and k >= 2:
                tw = g * g * M2t @ W[k - 2] + 2 * d   # crude offset accumulation
                one = (1 - a) * one + a * tw
            w[idx[k]] = one
    return f


if __name__ == "__main__":
    for lab, f in [("one-step x0 (=base)", mk(M1)), ("one-step fresh", mk(M1f)), ("avg x0/fresh", mk((M1 + M1f) / 2)),
                   ("avg one+two step", mk(M1, M2)), ("two-step weight .25", mk(M1, M2, .25))]:
        net, d = X.price(Var(f))
        errs = " ".join(f"{e / s:.2f}" for r, (e, s) in sorted(d["rowerr"].items()) if r.startswith("out"))
        print(f"{lab:24s} net {net:+.1f} gain {d['C'][0]:.3f} | {errs}")
