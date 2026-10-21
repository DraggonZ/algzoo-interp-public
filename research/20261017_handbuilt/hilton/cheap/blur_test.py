"""Development test of a floor fix: the seven smaller numbers vary, so the tie margin D varies around its mean-background
value. To first order D + tau Z, with tau^2 = sum over the seven background positions of (dD/dx_j)^2 times the
variance of a normal truncated below the third-largest. A thin band's width is |margin| / slope, so with the spread the
pair of orderings sharing a tie loses E|D + tau Z| / slope instead of |D| / slope. Here the slope is the effective one
from the exact node width (-D / width of the loser; development only), applied to both orderings; gross bands (width
>= 0.3) are left as they are. Does it move the band model toward the true accuracy?"""
import sys, numpy as np, torch
from scipy.stats import norm
import common as c, bandlib as bl
log = c.Log("blur_test.log")
names = sys.argv[1:] or ["ARC"] + [str(s) for s in c.SEEDS_DEV]
alln = [n for n, _ in c.nets()]; rows = []
idx = {tuple(t): i for i, t in enumerate(c.T.cpu().numpy().tolist())}
for name, W in c.nets():
    if name not in names: continue
    e = np.load(f"exact_node_{name}.npz"); ex = {"23": e["A"], "12": e["B"]}; blur = {}
    for tie in ("23", "12"):
        nd = bl.N23 if tie == "23" else bl.N12
        X, _ = bl.node_inputs(tie, c.T); X.requires_grad_(True)
        S, _ = c.scores(W, X); D, _, k = c.margin_from(S, c.T[:, 1])
        g, = torch.autograd.grad(D.sum(), X)
        bgmask = torch.ones_like(X, dtype=torch.bool); ar = torch.arange(720, device=c.DEV)
        for j in range(3): bgmask[ar, c.T[:, j]] = False
        u = nd[2]; r = norm.pdf(u) / norm.cdf(u); var = 1 - u * r - r * r
        tau = torch.sqrt(((g * bgmask) ** 2).sum(1) * var).detach().cpu().numpy(); D = D.detach().cpu().numpy()
        w = ex[tie].copy(); out = w.copy()
        # pair partner: same tie input, the two tied positions swapped
        T = c.T.cpu().numpy()
        for i, (p1, p2, p3) in enumerate(T):
            if D[i] >= 0 or w[i] >= 0.3 or w[i] <= 0: continue
            j = idx[(p1, p3, p2)] if tie == "23" else idx[(p2, p1, p3)]
            sig = -D[i] / w[i]; x = D[i] / max(tau[i], 1e-9)
            psi = lambda z: norm.pdf(z) - z * norm.cdf(-z)
            out[i] = tau[i] * psi(x) / sig                 # loser: E max(-(D + tau Z), 0) / slope
            if D[j] > 0 and w[j] == 0: out[j] = tau[i] * psi(-x) / sig   # winner gains a thin band
        blur[tie] = out
        log(f"   {name} tie {tie}: median tau {np.median(tau):.2f}, median |D| {np.median(np.abs(D)):.2f}")
    t = lambda x: torch.tensor(x, device=c.DEV, dtype=c.dt)
    a0 = bl.accuracy(t(ex["23"]), t(ex["12"])); a1 = bl.accuracy(t(blur["23"]), t(blur["12"])); tru = 100 * c.TRUE[alln.index(name)]
    rows.append((a0, a1, tru)); log(f"{name:>4s}: true {tru:.2f}; band model (exact node widths) {a0:.2f}; with background spread {a1:.2f}")
r = np.array(rows)
log(f"RMSE against truth over {len(r)} networks: without spread {np.sqrt(((r[:,0]-r[:,2])**2).mean()):.2f}, with {np.sqrt(((r[:,1]-r[:,2])**2).mean()):.2f}; mean error {np.mean(r[:,0]-r[:,2]):+.2f} / {np.mean(r[:,1]-r[:,2]):+.2f}")
