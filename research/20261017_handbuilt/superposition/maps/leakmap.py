"""Look 6, see README.md."""
import numpy as np, torch as th, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from algzoo.core import example
from algzoo.handbuilt_v23 import build_v23
NETS = {"ARC's network": [np.asarray(p, float) for p in example()],
        "Hand-built, 16 neurons": [np.asarray(p, float) for p in build_v23(U_mem=1.0, share=0.25, rho=0.99, level_exact=False).params]}
rng = np.random.default_rng(11); L = []
while len(L) < 400:
    x = rng.standard_normal(10); v = np.sort(x)[::-1]
    if v[1] > 0 and v[1] - v[2] < 0.05 * v[1]: L.append(x)
X = np.array(L); O = np.argsort(-X, 1)
def grads(P):
    wi, wh, wo = (th.tensor(p) for p in P); x = th.tensor(X, requires_grad=True); h = th.zeros(len(X), wi.shape[0], dtype=th.float64)
    for t in range(10): h = th.relu(h @ wh.T + x[:, t:t + 1] * wi)
    s = h @ wo.T; n = th.arange(len(X)); lead = s[n, th.tensor(O[:, 1])] - s[n, th.tensor(O[:, 2])]
    g, = th.autograd.grad(lead.sum(), x); return g.numpy(), lead.detach().numpy()
order_key = [("".join("LST"[k] for k in np.argsort(O[i, :3])), O[i, 0], O[i, 1], O[i, 2]) for i in range(len(X))]   # arrival order of Largest, Second, Third
rows = sorted(range(len(X)), key=lambda i: order_key[i])
LIM = 3.0   # shared colour limit (lead units per unit of the number), about ARC's 90th percentile
fig, axs = plt.subplots(1, 2, figsize=(9, 9), facecolor="white")
for ax, (n, P) in zip(axs, NETS.items()):
    g, lead = grads(P)
    sc = np.median(np.abs(X[:, None] * 0 + 1) * 0 + 1)    # placeholder to keep units raw
    G = g.copy(); G[np.arange(len(X))[:, None], O[:, :3]] = np.nan
    G = G[rows]
    small = np.abs(g[np.arange(len(X))[:, None], O[:, 3:]])
    print(f"{n}: |slope| of the lead on the seven smaller numbers: median {np.median(small):.3f}, 90th percentile {np.percentile(small, 90):.3f}; median |lead| {np.median(np.abs(lead)):.3f}")
    lim = LIM
    ax.imshow(G, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
    ax.set_title(f"{n}\n(colour limit ±{lim:.2f})", fontsize=10)
    ax.set_xticks(range(10)); ax.set_xlabel("position", fontsize=9)
    ks = [order_key[i][0] for i in rows]; starts = [0] + [k for k in range(1, len(ks)) if ks[k] != ks[k - 1]]
    ax.set_yticks(starts); ax.set_yticklabels([ks[s] for s in starts], fontsize=8)
    for s in starts[1:]: ax.axhline(s - 0.5, color="black", lw=0.6)
axs[0].set_ylabel("400 lists, grouped by arrival order (L largest, S second, T third)", fontsize=9)
fig.suptitle("Slope of the second-minus-third lead on each number (red: raising it helps the second; blue: the third; white: top three)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig("leakmap.png", dpi=160, facecolor="white"); print("saved")
