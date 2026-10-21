"""Gain or offset? (With a matched baseline.) Constructed lists:
largest L at pL, competitor S at pS, number x at px, -0.3 elsewhere. The comparison is score[px] - score[pS]
(no centring). For each S, find the x at which x overtakes S (scan of x, first crossing above 0.3 by linear
interpolation; several crossings are reported). Fair votes: x* = S. Fixed unequal gains: x* = r * S. An offset:
x* - S changes with S at a fixed L. Settings: gated (both after L), revealed (both before L), and the two mixed ones."""
import numpy as np

from algzoo.core import example

wi, wh, wo = example()


def scores(X):
    h = np.zeros((len(X), 16))
    for t in range(10):
        h = np.maximum(0, h @ wh.T + X[:, t:t + 1] * wi)
    return h @ wo.T


xs = np.linspace(0.3, 2.0, 1701)
settings = {"both after largest (L at 1, S at 3, x at 6)": (1, 3, 6), "both after, x first (L at 1, x at 3, S at 6)": (1, 6, 3),
            "both before largest (S at 2, x at 4, L at 7)": (7, 2, 4), "both before, x first (x at 2, S at 4, L at 7)": (7, 4, 2),
            "S before, x after (S at 2, L at 4, x at 7)": (4, 2, 7), "x before, S after (x at 2, L at 4, S at 7)": (4, 7, 2)}
for L in (2.0, 3.0):
    print(f"largest L = {L}")
    for name, (pL, pS, px) in settings.items():
        row = []
        for S in (0.4, 0.6, 0.8, 1.0, 1.2, 1.4):
            if S >= L:
                continue
            X = np.full((len(xs), 10), -0.3); X[:, pL] = L; X[:, pS] = S; X[:, px] = xs
            keep = xs < L
            sc = scores(X[keep]); m = sc[:, px] - sc[:, pS]; xk = xs[keep]
            cr = np.where(np.sign(m[1:]) != np.sign(m[:-1]))[0]
            roots = [xk[i] - m[i] * (xk[i + 1] - xk[i]) / (m[i + 1] - m[i]) for i in cr]
            row.append(f"S={S}: " + (",".join(f"{rt:.3f}" for rt in roots) if roots else "none"))
        print(f"  {name}: x* = " + " | ".join(row))

print("\nDoes the constant offset come from the filler numbers? Offset x* - S at S = 1.0, L = 2.0, by filler value.")
for name, (pL, pS, px) in settings.items():
    out = []
    for bg in (-0.05, -0.15, -0.3, -0.6, -1.0):
        X = np.full((len(xs), 10), bg); X[:, pL] = 2.0; X[:, pS] = 1.0; X[:, px] = xs
        keep = xs < 2.0; sc = scores(X[keep]); m = sc[:, px] - sc[:, pS]; xk = xs[keep]
        cr = np.where(np.sign(m[1:]) != np.sign(m[:-1]))[0]
        roots = [xk[i] - m[i] * (xk[i + 1] - xk[i]) / (m[i + 1] - m[i]) for i in cr]
        out.append(f"{bg}: " + (",".join(f"{rt - 1.0:+.4f}" for rt in roots) if roots else "none"))
    print(f"  {name}: " + " | ".join(out))
