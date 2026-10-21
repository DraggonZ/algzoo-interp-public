"""How wide is the uncertain zone? For the second-versus-third decision, the share of lists where the network names the
third as a function of the gap g = (second - third) / second, like a psychometric curve. Fit P(third) = Phi((m - g) / w):
m is the lean (offset of the boundary), w the width. Same for the largest-versus-second decision with
h = (largest - second) / largest. Then per pair of positions (second, third): spread of the fitted widths and leans."""
import sys; sys.path.insert(0, "../steer")
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from harness import *
X = lists(16_000_000, seed=501); o = X.argsort(1, descending=True); v = X.sort(1, descending=True).values
a = run(ARC, X).argmax(1); m = v[:, 1] > 0.1
g = ((v[:, 1] - v[:, 2]) / v[:, 1])[m].cpu().numpy(); y3 = (a == o[:, 2])[m].cpu().numpy()
h = ((v[:, 0] - v[:, 1]) / v[:, 0])[m].cpu().numpy(); yL = (a == o[:, 0])[m].cpu().numpy()
P, Q = o[:, 1][m].cpu().numpy(), o[:, 2][m].cpu().numpy()
def fit(x, y, sel=None):
    if sel is not None: x, y = x[sel], y[sel]
    k = x < 0.4; x, y = x[k], y[k]
    nll = lambda p: -np.sum(y * norm.logcdf((p[0] - x) / np.exp(p[1])) + (1 - y) * norm.logcdf(-(p[0] - x) / np.exp(p[1])))
    r = minimize(nll, [0.0, np.log(0.03)], method="Nelder-Mead"); return r.x[0], np.exp(r.x[1])
for lab, x, y in (("third beats second, gap (2nd-3rd)/2nd", g, y3), ("largest beats second, gap (1st-2nd)/1st", h, yL)):
    mu, w = fit(x, y); print(f"{lab}: lean {mu:+.4f}, width {w:.4f}")
    bins = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
    print("   observed / fitted by gap bin: " + " ".join(f"{lo:.3f}-{hi:.3f}: {100*y[(x >= lo) & (x < hi)].mean():.1f}/{100*norm.cdf((mu - (lo + hi) / 2) / w):.1f}" for lo, hi in zip(bins, bins[1:])))
ws, ms = [], []
for p in range(10):
    for q in range(10):
        if p == q: continue
        sel = (P == p) & (Q == q)
        mu, w = fit(g, y3, sel); ws.append(w); ms.append(mu)
ws, ms = np.array(ws), np.array(ms)
print(f"per position pair (90): width median {np.median(ws):.4f} (10-90%: {np.quantile(ws, 0.1):.4f}-{np.quantile(ws, 0.9):.4f}); lean median {np.median(ms):+.4f} (10-90%: {np.quantile(ms, 0.1):+.4f} to {np.quantile(ms, 0.9):+.4f})")
np.savez("curve.npz", ws=ws, ms=ms)
