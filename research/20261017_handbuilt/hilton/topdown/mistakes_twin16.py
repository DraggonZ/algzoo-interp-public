"""Hand-built only (run with --handbuilt). Reference for 'mistakes reproduced': how often do independently trained networks of the same size (run 3, 32 seeds)
and the hand-built networks make ARC's wrong choice on the lists where ARC's network is wrong? Grid 40 x 48 x 48, no
sampling; split into thirds of the mistakes' probability by ARC's wrong answer's lead (as in mistakes_by_margin.py)."""
import numpy as np, torch
import store_ladder as sl
from algzoo.core import load
from algzoo.handbuilt_v21 import build_v21
from algzoo.handbuilt_v18 import build_v18
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v23 import build_v23

DEV = "cuda"; dt = torch.float64
def T(W): return [torch.tensor(np.asarray(p, float), device=DEV, dtype=dt) for p in W]
def net(W):
    def f(X):
        h = torch.zeros(len(X), len(W[0]), device=DEV, dtype=dt)
        for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
        return h @ W[2].T
    return f

if __name__ == "__main__":
    pass
    models = [("twin16 (v23, ARC constants)", net(T(build_v23(U_mem=1.0, share=0.25, rho=0.99, level_exact=False).params))), ("v19 (16)", net(T(build_v19(kappa=0.5).params))), ("v21 (28 neurons)", net(T(build_v21())))]
    import sys
    if "--handbuilt" in sys.argv: models = models[:3]
    seeds = list(range(100, 116)) + list(range(200, 216))
    models += [(f"trained seed {s}", net(T(load(f"/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_{s}.pth")))) for s in seeds]
    nV = len(sl.wtt); rec = {k: [] for k, _ in models}; right = {k: [0.0, 0.0] for k, _ in models}; M, Wt = [], []
    for i in range(0, 720, 16):
        tt = sl.full_mean.T[i:i + 16]; pos = tt.repeat_interleave(nV, 0); val = sl.Vt.repeat(len(tt), 1); n = len(pos)
        bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
        X = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
        for k in range(3): X[ar, pos[:, k]] = val[:, k]
        w = sl.wtt.repeat(len(tt)); y = pos[:, 1]; S0 = sl.run(X); a0 = S0.argmax(1); wr = a0 != y
        M.append((S0[ar, a0] - S0[ar, y])[wr].cpu().numpy()); Wt.append(w[wr].cpu().numpy())
        for k, f in models:
            a1 = f(X).argmax(1); rec[k].append((a1 == a0)[wr].cpu().numpy())
            right[k][0] += (w * (a1 == y)).sum().item(); right[k][1] += w.sum().item()
    M, Wt = np.concatenate(M), np.concatenate(Wt); o = np.argsort(M); cw = np.cumsum(Wt[o]) / Wt.sum()
    band = np.digitize(M, [M[o][np.searchsorted(cw, q)] for q in (1 / 3, 2 / 3)])
    print(f"{'network':26s} {'grid acc':>8s} {'all':>6s} {'coin':>6s} {'mid':>6s} {'clear':>6s}")
    rows = []
    for k in rec:
        r = np.concatenate(rec[k]); vals = [100 * (Wt * r).sum() / Wt.sum()] + [100 * (Wt * r)[band == b].sum() / Wt[band == b].sum() for b in range(3)]
        rows.append(vals); print(f"{k:26s} {100*right[k][0]/right[k][1]:7.2f}% " + " ".join(f"{v:5.1f}%" for v in vals))
    R = np.array(rows[2:])
    print("32 trained networks, median [min, max]: " + ", ".join(f"{n} {np.median(R[:, j]):.1f} [{R[:, j].min():.1f}, {R[:, j].max():.1f}]" for j, n in enumerate(("all", "coin", "mid", "clear"))))
