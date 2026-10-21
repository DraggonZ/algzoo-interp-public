"""r13b look 3: the push's common offset from the readout's zero column sums.
Softmax ignores a score added to every position, so fair votes fix each row only up to a common offset d (one
16-vector, the same for every k): row_k = g Push(row_{k-1}) + d, k = 1..9. Summing over k = 1..9 and using zero
column sums (sum_{k=0..9} row_k = 0):  -row_0 = g Push(-row_9) + 9 d,  so  d = (g Push(row_9) - row_0) / 9.
Variants priced by the explorer (gain refitted). Rows 0 and 9 must be uncovered or described before rows 1-8."""
import numpy as np
import torch as th
import r12a_x as X

R = X.load("rules/r12a_regpush.py")
WI, WH, WO = X.WI, X.WH, X.WO
N = 400_000
rg = np.random.default_rng(101)
x = rg.standard_normal((N, 10))
h = np.zeros((N, 16))
for t in range(10):
    h = np.maximum(h @ WH.T + x[:, t, None] * WI[None, :], 0)
h11 = np.maximum(h @ WH.T, 0)
M = np.linalg.solve(h.T @ h / N, h.T @ h11 / N)
assert np.allclose(M @ WO[0], R._REG[0])
P9 = M @ WO[9]
print("|P row9|/|row0| %.2f, cos(P row9, row0) %+.2f" % (np.linalg.norm(P9) / np.linalg.norm(WO[0]),
      P9 @ WO[0] / np.linalg.norm(P9) / np.linalg.norm(WO[0])))
E = WO[1:9] - 1.0177 * R._REG
dstar = (P9 - WO[0]) / 9
print("cos(d*, mean residual) %+.2f; |d*|/|mean res| %.2f" % (dstar @ E.mean(0) / np.linalg.norm(dstar) / np.linalg.norm(E.mean(0)),
      np.linalg.norm(dstar) / np.linalg.norm(E.mean(0))))

names = list(R.COVERED)


class Var:
    def __init__(self, mode, drop9=False, push9=False, extra=0):
        self.mode, self.push9 = mode, push9
        keep = [n for n in names if not (drop9 and n.endswith("->out9"))]
        self.keep_idx = th.tensor([names.index(n) for n in keep])
        self.COVERED = keep
        self.ORDER = [o for o in R.ORDER if not (drop9 and o == "out9")]
        self.KINDS = R.KINDS; self.init_constants = R.init_constants
        self.STRUCTURE_BITS = R.STRUCTURE_BITS + extra
        self.pos = {n: i for i, n in enumerate(keep)}
        self.M = th.tensor(M); self.W0 = th.tensor(WO[0]); self.W9 = th.tensor(WO[9]); self.W8 = th.tensor(WO[8])

    def weights(self, C):
        w = R.weights(C)[self.keep_idx].clone()
        g = C[0]
        if self.mode == "c19":
            d = -self.W0 / 9
        elif self.mode == "zs":
            d = (g * self.M @ self.W9 - self.W0) / 9
        else:
            d = th.zeros(16, dtype=th.float64)
        rows = list(range(1, 9))
        out = []
        for n in self.COVERED:
            out.append(None)
        w = list(w)
        for k in rows:
            for i in range(16):
                j = self.pos[f"{i}->out{k}"]
                w[j] = w[j] + d[i]
        if self.push9:
            p = g * self.M @ self.W8 + d
            for i in range(16):
                n = f"{i}->out9"
                if n in self.pos:
                    w[self.pos[n]] = p[i]
        return th.stack(w)


for label, kw in [("base", dict(mode="none")),
                  ("base, row 9 uncovered", dict(mode="none", drop9=True)),
                  ("c = 1/9, row 9 zero-sum", dict(mode="c19")),
                  ("c = 1/9, row 9 uncovered", dict(mode="c19", drop9=True)),
                  ("d from zero sums (uses ARC row 9), row 9 uncovered", dict(mode="zs", drop9=True)),
                  ("d from zero sums, row 9 zero-sum [circular, diagnostic]", dict(mode="zs")),
                  ("d from zero sums, row 9 pushed too [circular, diagnostic]", dict(mode="zs", push9=True))]:
    net, dd = X.price(Var(**kw))
    errs = " ".join(f"{r}:{e / s:.2f}" for r, (e, s) in sorted(dd["rowerr"].items()) if r.startswith("out"))
    print(f"{label:60s} net {net:+.1f} gain {dd['C'][0]:.3f} rest {dd['rest']:.0f} struct {dd['struct']} | {errs}")
