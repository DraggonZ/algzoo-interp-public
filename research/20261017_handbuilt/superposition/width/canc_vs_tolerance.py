"""Across trained networks: does tolerance to small errors in the final state follow from per-neuron cancellation?
Cancellation: median over random lists of (sum over neurons of |push| on second-minus-third) / |decision| (as the outer
review's canc.py). Tolerance: accuracy drop when Gaussian noise of 0.5% of each neuron's variance is added to the final
state only. Also the prediction 'drop ~ cancellation x noise size' via the share of lists whose decision is smaller than
the noise it receives. Networks: ARC, run-3 seeds 100-115 and 200-215, bo5 16, 20, 32 bo5, 32 long."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family")
from common import *
from algzoo.core import load
M = "/data/output/research/algzoo/models/extra/"
if __name__ == "__main__":
    nets = {"ARC": ARC, **{f"run3 {s}": trained(s) for s in list(range(100, 116)) + list(range(200, 216))},
            "16 bo5": T(load(M + "2nd_argmax_16_10_1073741824_bo5.pth")), "20": T(load(M + "2nd_argmax_20_1073741824.pth")),
            "32 bo5": T(load(M + "2nd_argmax_32_10_1073741824_bo5.pth")), "32 long": T(load(M + "2nd_argmax_32_17179869184.pth"))}
    g = torch.Generator(device=DEV).manual_seed(5); X = torch.randn(200000, 10, device=DEV, dtype=dt, generator=g); L, S, Tq = top3(X); ar = torch.arange(len(X), device=DEV)
    rows = []
    for nm, W in nets.items():
        h = torch.zeros(len(X), len(W[0]), device=DEV, dtype=dt); hs = []
        for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0]); hs.append(h)
        sd = torch.stack(hs, 1).reshape(-1, len(W[0])).std(0)
        P = h * (W[2][S] - W[2][Tq]); canc = (P.abs().sum(1) / P.sum(1).abs().clamp_min(1e-9)).median().item()
        acc = ((h @ W[2].T).argmax(1) == S).double().mean().item()
        gg = torch.Generator(device=DEV).manual_seed(1)
        hn = h + torch.randn(h.shape, device=DEV, dtype=dt, generator=gg) * sd * np.sqrt(0.005)
        accn = ((hn @ W[2].T).argmax(1) == S).double().mean().item()
        rows.append((nm, len(W[0]), canc, acc, acc - accn)); print(f"{nm:10s} {len(W[0]):2d} neurons: cancellation {canc:6.1f}, accuracy {100*acc:.2f}%, drop with 0.5% noise on the final state {100*(acc-accn):5.1f} points", flush=True)
    c = np.array([r[2] for r in rows]); dr = np.array([r[4] for r in rows])
    from scipy.stats import spearmanr
    print(f"rank correlation between cancellation and the drop, all {len(rows)} networks: {spearmanr(c, dr).correlation:.2f}; 16-neuron networks only: {spearmanr(c[[r[1]==16 for r in rows]], dr[[r[1]==16 for r in rows]]).correlation:.2f}")
