"""What sets the second-third near-tie band in each placement? (diagnostic, deterministic)

Background b = -0.46 for the seven small inputs. For each ordered placement (p1, p2, p3), m2 = 0.9, m1 = m2 + g12:
  measured band: length of the wrong set for m3 in [m2 - 0.3, m2], scanned in steps of 0.001;
  tie margin M0: score[p2] - score[p3] at m3 = m2 (centered scores);
  leftover difference dS: the same difference when the largest is alone on the background (S1[p2] - S1[p3]), which depends
  only on p1 and m1 (10 runs per m1).
Hypothesis: the band is the largest's uncancelled leftover, so M0 ~ dS and band ~ max(0, -M0) / slope.
"""
import itertools
import sys

import numpy as np
import torch as th

from algzoo.core import example, zoo

dev = "cuda"
b = -0.46
T = np.array(list(itertools.permutations(range(10), 3))); n = len(T); ar = np.arange(n)
G23 = np.arange(0, 0.3001, 0.001)


def scores(params, X):
    w_hi, w_hh, w_oh = (th.as_tensor(q, dtype=th.float64, device=dev) for q in params)
    x = th.as_tensor(X, device=dev)
    h = th.zeros(x.shape[0], len(w_hi), dtype=th.float64, device=dev)
    for t in range(10):
        h = th.relu(h @ w_hh.T + x[:, t, None] * w_hi)
    s = h @ w_oh.T
    return (s - s.mean(1, keepdim=True)).cpu().numpy()


def main():
  for name in sys.argv[1:] or ["example", 2, 3, 0, 4, 1]:
    p = example() if name == "example" else zoo(16, 10, int(name))
    m2 = 0.9
    for g12 in (0.2, 0.5, 1.0):
        m1 = m2 + g12
        # measured band
        X = np.full((n, len(G23), 10), b)
        X[ar, :, T[:, 0]] = m1; X[ar, :, T[:, 1]] = m2
        X[ar[:, None], np.arange(len(G23))[None, :], T[:, 2][:, None]] = m2 - G23[None, :]
        S = scores(p, X.reshape(-1, 10)).reshape(n, len(G23), 10)
        wrong = S.argmax(-1) != T[:, 1:2]
        band = wrong[:, 1:].sum(1) * 0.001          # wrong length within (0, 0.3]
        # tie margin and slope just off the tie
        M0 = S[ar, 0, T[:, 1]] - S[ar, 0, T[:, 2]]
        M5 = S[ar, 5, T[:, 1]] - S[ar, 5, T[:, 2]]
        slope = np.median((M5 - M0) / 0.005)
        # largest alone
        X1 = np.full((10, 10), b); X1[np.arange(10), np.arange(10)] = m1
        S1 = scores(p, X1)
        dS = S1[T[:, 0], T[:, 1]] - S1[T[:, 0], T[:, 2]]
        pred = np.maximum(0, -M0) / slope
        predL = np.maximum(0, -dS) / slope
        has = band > 0
        print(f"{name} g12={g12}: placements with a band {has.mean():.2f}, mean band {band.mean():.4f}; median slope {slope:.0f}; "
              f"corr(M0, dS) {np.corrcoef(M0, dS)[0, 1]:.2f}, M0 - dS median {np.median(M0 - dS):+.2f} (M0 spread {M0.std():.2f}); "
              f"mean band predicted from tie margin {pred.mean():.4f} (corr {np.corrcoef(pred, band)[0, 1]:.2f}), "
              f"from leftover alone {predL.mean():.4f} (corr {np.corrcoef(predL, band)[0, 1]:.2f})", flush=True)


if __name__ == "__main__":
    main()
