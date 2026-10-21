"""Frozen evaluation (settings fixed 2026-10-19 before this run: TLQ m = 1 and 4, Sobol seed 0, piece cap 40).
Networks: ARC and trained seeds 100-115, 200-215 (truth: hilton/propagate/across.npy column 0, 10M lists).
Noisy copies of ARC at s = 0.0025 and 0.005 (100 each; truth: level2/truth/noisy_truth.npz on 2M lists).
For each: TLQ estimate, its operation count, plain-sampling standard error at the same count, and the control:
quasi-random sampling of whole lists (scrambled Sobol, seed 100) at the same count. Writes evaluate.npz."""
import sys, numpy as np, torch, time
sys.path.insert(0, "../../../family")
from common import ARC, trained, scores
import tlq
from linelib import F_LIST, DEV, dt
tr = np.load("../../../hilton/propagate/across.npy")[:, 0]
names = ["ARC"] + [str(s) for s in list(range(100, 116)) + list(range(200, 216))]
nets = [(n, ARC if n == "ARC" else trained(int(n)), tr[i]) for i, n in enumerate(names)]
nz = np.load("../truth/noisy_truth.npz"); wi, wh, wo = ARC
for s in (0.0025, 0.005):
    for c, v in enumerate(nz[f"dw_{s}"]):
        v = torch.tensor(v, device=DEV, dtype=dt)
        nets.append((f"s{s}#{c}", [wi + v[:16], wh + v[16:272].reshape(16, 16), wo + v[272:].reshape(10, 16)], nz[f"stats_{s}"][c, 0]))
def qmc(W, N, seed=100):
    eng = torch.quasirandom.SobolEngine(10, scramble=True, seed=seed)
    X = torch.special.ndtri(eng.draw(N).to(DEV, dt).clamp(1e-12, 1 - 1e-12))
    return np.mean([(scores(W, X[i:i + 500000]).argmax(1) == X[i:i + 500000].argsort(1, descending=True)[:, 1]).double().mean().item() * len(X[i:i + 500000]) for i in range(0, N, 500000)]) * 1.0
rows = []; t0 = time.time()
for name, W, t in nets:
    row = [t]
    for m in (1, 4):
        a, fl = tlq.estimate(W, m, 0); N = int(round(fl / F_LIST))
        X = None
        eng = torch.quasirandom.SobolEngine(10, scramble=True, seed=100)
        Xq = torch.special.ndtri(eng.draw(N).to(DEV, dt).clamp(1e-12, 1 - 1e-12))
        aq = (scores(W, Xq).argmax(1) == Xq.argsort(1, descending=True)[:, 1]).double().mean().item()
        row += [a, fl, aq]
    rows.append(row)
    if not name.startswith("s") or name.endswith("#0") or name.endswith("#99"):
        print(f"{name}: true {100*t:.2f}  TLQ m1 {100*row[1]:.2f} ({row[2]/F_LIST:.0f} lists)  m4 {100*row[4]:.2f} ({row[5]/F_LIST:.0f})  quasi-random lists {100*row[3]:.2f} / {100*row[6]:.2f}  {time.time()-t0:.0f}s", flush=True)
R = np.array(rows); np.savez("evaluate.npz", R=R, names=np.array([n for n, _, _ in nets]))
groups = {"ARC": [0], "development 200-215": list(range(17, 33)), "held out 100-115": list(range(1, 17)),
          "noisy s 0.0025": list(range(33, 133)), "noisy s 0.005": list(range(133, 233))}
print("\nRMSE in points (one deterministic estimate per network); sampling = sqrt(mean p(1-p) / n) at TLQ's operation count")
for g, ix in groups.items():
    t = R[ix, 0]
    for m, (ca, cf, cq) in ((1, (1, 2, 3)), (4, (4, 5, 6))):
        e = 100 * np.sqrt(np.mean((R[ix, ca] - t) ** 2)); n = R[ix, cf] / F_LIST
        samp = 100 * np.sqrt(np.mean(t * (1 - t) / n)); eq = 100 * np.sqrt(np.mean((R[ix, cq] - t) ** 2))
        bias = 100 * np.mean(R[ix, ca] - t)
        print(f"  {g:22s} m {m}: TLQ {e:.3f} (mean error {bias:+.3f}), cost {n.mean():.0f} list-equivalents = {R[ix, cf].mean():.3g} operations; "
              f"plain sampling {samp:.3f}; quasi-random lists {eq:.3f}")
