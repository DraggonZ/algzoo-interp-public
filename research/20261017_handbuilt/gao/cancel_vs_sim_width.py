"""Does cancellation predict how hard neurons are to simulate beyond 16-neuron networks? AlgZoo second-largest networks
for ten numbers with 8, 16 and 32 neurons (five seeds each, trained on 2^30 sequences; those with at least 80% accuracy).
Per network: lead / pushes, and the accuracy lost when one neuron (median and worst over neurons that are ever active)
is replaced by its formula from the same role-free 30-quantity menu as sim_across.py (least squares, fit 50,000 lists
seed 101, scored on 50,000 seed 202). Rank correlations over networks."""
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from algzoo.core import load
import simlib
from simlib import lists
from sim_across import feats

Xf, Xs = lists(101, 50_000), lists(202, 50_000); Ff, Fs = feats(Xf), feats(Xs); A = Ff.reshape(-1, Ff.shape[-1])
y = Xs.argsort(1)[:, -2]; r = np.arange(len(Xs)); rows = []
for d in (8, 16, 32):
    for f in sorted(Path("/data/output/research/algzoo/models/zoo").glob(f"2nd_argmax_{d}_10_1073741824_*.pth")):
        P = tuple(np.asarray(p, float) for p in load(str(f))); P = (P[0].reshape(-1), P[1], P[2])
        _, Pf, _ = simlib.run(Xf, params=P); Ls, _, Hs = simlib.run(Xs, params=P); acc0 = (Ls.argmax(1) == y).mean()
        if acc0 < 0.8:
            continue
        k = Ls.argmax(1); j = np.argsort(Ls, 1)[:, -2]
        ratio = np.median((Ls[r, k] - Ls[r, j]) / np.maximum(np.abs((P[2][k] - P[2][j]) * Hs[:, 9]).sum(1), 1e-12))
        losses = []
        for n in range(d):
            if Hs[:, :, n].max() <= 0:
                continue
            co = np.linalg.lstsq(A, Pf[:, :, n].reshape(-1), rcond=None)[0]
            losses.append(acc0 - (simlib.run(Xs, {n: np.maximum(Fs @ co, 0)}, params=P)[0].argmax(1) == y).mean())
        rows.append((d, ratio, np.median(losses), max(losses), acc0))
        print(f"{f.stem}: accuracy {acc0:.3f}; lead / pushes {ratio:.4f}; loss, one neuron replaced: median {np.median(losses):.3f}, worst {max(losses):.3f}", flush=True)
R = np.array(rows)
for lab, i in (("median", 2), ("worst", 3)):
    rho, p = spearmanr(R[:, 1], R[:, i]); print(f"all widths ({len(R)} networks): lead/pushes vs {lab} single-neuron loss {rho:.2f} (p {p:.4f})")
