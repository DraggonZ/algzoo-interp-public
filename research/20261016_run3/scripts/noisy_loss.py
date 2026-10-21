"""When a noisy copy loses the vote while sliding, what wins? And within-noise-level correlations.

For each copy and each two-spike case (90 pairs x ratios 0.3, 0.7, 0.95) where the vote was placed right but the final
answer is wrong: is the winning slot the one the largest-only input would pick (its leftover vote)? Also: does the
second's own vote (scores with both minus scores with the largest only) still point at the right slot at the end?
"""
import numpy as np
from scipy.stats import spearmanr

from algzoo.bench import load_bench

names, acc, models = load_bench(("mult",))
R = np.load("/data/output/research/algzoo/20261016_run3/reports/noisy_mechanism.npy")
sig = np.array([float(str(n).split("_")[0][4:]) for n in names])


def run_scores(params, x):
    w_hi, w_hh, w_oh = params
    h = np.zeros(len(w_hi)); S = []
    for t in range(10):
        h = np.maximum(0, w_hh @ h + x[t] * w_hi)
        s = w_oh @ h; S.append(s - s.mean())
    return np.array(S)


tot = dict(lost=0, winner_is_leftover=0, own_vote_right=0, winner_9=0)
for nm, p in zip(names, models):
    for r in (0.3, 0.7, 0.95):
        for p1 in range(10):
            for p2 in range(10):
                if p1 == p2:
                    continue
                x = np.zeros(10); x[p1] = 1.0; x[p2] = r
                S = run_scores(p, x)
                t0 = max(p1, p2)
                if S[t0].argmax() != p2 + (9 - t0) or S[9].argmax() == p2:
                    continue
                x1 = np.zeros(10); x1[p1] = 1.0
                S1 = run_scores(p, x1)
                tot["lost"] += 1
                tot["winner_is_leftover"] += S[9].argmax() == S1[9].argmax()
                tot["own_vote_right"] += (S[9] - S1[9]).argmax() == p2
                tot["winner_9"] += S[9].argmax() == 9
print(f"cases placed right but lost while sliding: {tot['lost']}")
print(f"  winner = slot the largest alone would pick: {tot['winner_is_leftover'] / tot['lost']:.1%}")
print(f"  winner = slot 9: {tot['winner_9'] / tot['lost']:.1%}")
print(f"  the second's own vote still points at its slot at the end: {tot['own_vote_right'] / tot['lost']:.1%}")
lab = ["accuracy", "Z2", "placed", "held", "lost sliding", "slide", "lone spread", "ring miss", "ring base"]
for s in (0.005, 0.01, 0.02):
    k = sig == s
    print(f"sigma {s}: Spearman with Z2 — " + ", ".join(f"{lab[j]} {spearmanr(R[k, 1], R[k, j])[0]:+.2f}" for j in (3, 4, 5, 6, 7)))
