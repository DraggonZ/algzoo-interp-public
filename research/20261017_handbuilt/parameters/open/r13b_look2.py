"""r13b look 2: offsets added to r12a's pushed readout rows 1-8, priced by the explorer (exact v2 net, gain refitted).
(a) d = -c row_0 on all rows (r12b's regularity) - reference only.
(b) d from the readout's zero column sums, with the rows iterated from row 0 through the rule's own push:
    p_0 = row_0, p_k = Push(p_{k-1}); d = -(sum_{k=0..9} p_k)/9 (the common offset that makes rows 0..9 sum to zero).
    Uses only row 0 (uncovered) and the network's states. Diagnostic."""
import numpy as np
import torch as th
import r12a_x as X

R = X.load("rules/r12a_regpush.py")
WI, WH, WO = X.WI, X.WH, X.WO
names = list(R.COVERED)
idx = {k: [names.index(f"{i}->out{k}") for i in range(16)] for k in range(1, 9)}


class Off:
    def __init__(self, D, extra_struct=0):
        self.D = np.asarray(D)  # [8,16] added to rows 1-8
        self.COVERED = R.COVERED; self.ORDER = R.ORDER; self.KINDS = R.KINDS
        self.STRUCTURE_BITS = R.STRUCTURE_BITS + extra_struct
        self.init_constants = R.init_constants
        add = np.zeros(len(names))
        for k in range(1, 9):
            add[idx[k]] = self.D[k - 1]
        self.add = th.tensor(add, dtype=th.float64)

    def weights(self, C):
        return R.weights(C) + self.add


base, _ = X.price(Off(np.zeros((8, 16))))
print(f"base {base:+.1f}")
for c in [0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.2]:
    net, d = X.price(Off(-c * np.tile(WO[0], (8, 1))))
    print(f"(a) c={c:.2f}: net {net:+.1f} gain {d['C'][0]:.3f}")

# the push operator as a matrix: row_k = M row_{k-1}, from the rule's regression (states only)
N = 400_000
g = np.random.default_rng(101)
x = g.standard_normal((N, 10))
h = np.zeros((N, 16))
for t in range(10):
    h = np.maximum(h @ WH.T + x[:, t, None] * WI[None, :], 0)
h11 = np.maximum(h @ WH.T, 0)
G = h.T @ h / N
M = np.linalg.solve(G, h.T @ h11 / N)  # row_k = M @ row_{k-1}
assert np.allclose(M @ WO[0], R._REG[0])
p = [WO[0]]
for k in range(1, 10):
    p.append(M @ p[-1])
p = np.array(p)
S = p.sum(0)
d = -S / 9
print("iterated rows' sum vs row 0: cos %.2f, |sum|/|row0| %.2f" % (S @ WO[0] / np.linalg.norm(S) / np.linalg.norm(WO[0]),
      np.linalg.norm(S) / np.linalg.norm(WO[0])))
print("ARC rows sum |.|/|row0| %.3f" % (np.linalg.norm(WO.sum(0)) / np.linalg.norm(WO[0])))
for k in range(10):
    print(k, "iterated vs ARC rel err %.2f" % (np.linalg.norm(p[k] - WO[k]) / np.linalg.norm(WO[k])))
for s in [0.5, 1.0]:
    net, dd = X.price(Off(s * np.tile(d, (8, 1))))
    print(f"(b) d x {s}: net {net:+.1f} gain {dd['C'][0]:.3f}")
# residual's common part: least squares on ARC (reference, unpriced)
E = WO[1:9] - 1.0177 * R._REG
print("mean residual cos row0 %.2f, |mean|/|row0| %.3f" % (E.mean(0) @ WO[0] / np.linalg.norm(E.mean(0)) / np.linalg.norm(WO[0]),
      np.linalg.norm(E.mean(0)) / np.linalg.norm(WO[0])))
print("per-row coeff on row 0:", (E @ WO[0] / (WO[0] @ WO[0])).round(3))
print("cos(d, mean residual) %.2f" % (d @ E.mean(0) / np.linalg.norm(d) / np.linalg.norm(E.mean(0))))
net, dd = X.price(Off(np.tile(E.mean(0), (8, 1))))
print(f"mean residual (unpriced ceiling): {net:+.1f}")
