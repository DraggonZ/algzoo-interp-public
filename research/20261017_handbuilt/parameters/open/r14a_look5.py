"""r14a look 5: arrivals that cannot matter to the answer. The push's extra step with x_new drawn from N(0,1) truncated
below the list's smallest number (the newcomer is the smallest of all), x_new = the list's smallest, and fixed very
negative arrivals. Seeds 101, 102."""
import numpy as np
from scipy.stats import norm
import r12a_x as X
from r14a_look1 import Var, mk, WH, WI

N = 400_000
for seed in (101, 102):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, 11))
    u = rng.uniform(size=N)
    h = np.zeros((N, 16))
    for t in range(10):
        h = np.maximum(h @ WH.T + x[:, t, None] * WI[None, :], 0)
    G = h.T @ h / N
    pre = h @ WH.T
    mn = x[:, :10].min(1)

    def M(xn):
        return np.linalg.solve(G, h.T @ np.maximum(pre + xn[:, None] * WI[None, :], 0) / N)
    out = []
    for lab, xn in [("-|x|", -np.abs(x[:, 10])), ("below min", norm.ppf(u * norm.cdf(mn))), ("= min", mn),
                    ("-1", -np.ones(N)), ("-2", -2 * np.ones(N)), ("-5", -5 * np.ones(N))]:
        net, d = X.price(Var(mk(M(xn))))
        out.append(f"{lab} {net:+.1f} (g {d['C'][0]:.3f})")
    print(seed, " | ".join(out), flush=True)
