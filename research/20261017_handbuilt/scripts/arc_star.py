"""Build ARC*: the smallest change to ARC's weights that makes every close-call tie exactly even.
Ties (background -0.46): second = third = 1.0 with the largest g in GRID at each of the 720 placements
(margin score_p3 - score_p2), and largest = second = 1.6 with the third at 0.5 (margin score_p1 - score_p2).
Loss = mean(margin^2) / mean(ARC margin^2) + mu * mean((scores - ARC scores)^2) / mean(ARC scores^2) on 20,000
fixed random lists (keeps the mechanism: without it the optimizer shrinks every vote and accuracy falls to 68%)
+ lam * mean(((W - W_ARC) / rms(W_ARC per matrix))^2).
A mechanistic target (exact cancellation at ties), not accuracy. Saves arc_star_<lam>.pt and prints, per lam:
remaining tie margins, relative weight change, and the accuracy of ARC* on 2 million random lists."""
import itertools
import sys

import numpy as np
import torch as th

from algzoo.core import example

th.set_default_dtype(th.float64)
GRID = [1.05, 1.15, 1.3, 1.6, 2.0, 3.0, 5.0]
pl = np.array(list(itertools.permutations(range(10), 3))); N = len(pl); r = np.arange(N)


def tie_lists_random(n=40_000, seed=2):
    """Real close calls: random lists with the third-largest set equal to the second (n/2) or the second set equal
    to the largest (n/2); everything else stays random. Margin: the lower-ranked tied position minus the other."""
    g = th.Generator().manual_seed(seed); X = th.randn(n, 10, generator=g)
    X = X[(X > 0).sum(1) >= 3][: n // 1]
    o = X.argsort(1, descending=True); i = th.arange(len(X)); h = len(X) // 2
    a = th.cat([o[:h, 1], o[h:, 1]]); b = th.cat([o[:h, 2], o[h:, 0]])
    X[i[:h], o[:h, 2]] = X[i[:h], o[:h, 1]]; X[i[h:], o[h:, 1]] = X[i[h:], o[h:, 0]]
    return X, a, b


def tie_lists():
    Xs, A, B = [], [], []
    for g in GRID:
        X = np.full((N, 10), -0.46); X[r, pl[:, 0]] = g; X[r, pl[:, 1]] = 1.0; X[r, pl[:, 2]] = 1.0
        Xs.append(X); A.append(pl[:, 1]); B.append(pl[:, 2])
    X = np.full((N, 10), -0.46); X[r, pl[:, 0]] = 1.6; X[r, pl[:, 1]] = 1.6; X[r, pl[:, 2]] = 0.5
    Xs.append(X); A.append(pl[:, 1]); B.append(pl[:, 0])
    return th.tensor(np.concatenate(Xs)), th.tensor(np.concatenate(A)), th.tensor(np.concatenate(B))


def margins(W, X, a, b):
    wi, wh, wo = W
    h = th.zeros(len(X), len(wi))
    for t in range(10):
        h = th.relu(h @ wh.T + X[:, t:t + 1] * wi)
    s = h @ wo.T; i = th.arange(len(X))
    return s[i, b] - s[i, a]


def scores(W, X):
    wi, wh, wo = W; h = th.zeros(len(X), len(wi))
    for t in range(10):
        h = th.relu(h @ wh.T + X[:, t:t + 1] * wi)
    s = h @ wo.T
    return s - s.mean(1, keepdim=True)


def accuracy(W, n=2_000_000, seed=0):
    g = th.Generator().manual_seed(seed); X = th.randn(n, 10, generator=g)
    wi, wh, wo = W; h = th.zeros(n, len(wi))
    for t in range(10):
        h = th.relu(h @ wh.T + X[:, t:t + 1] * wi)
    return ((h @ wo.T).argmax(1) == X.argsort(1, descending=True)[:, 1]).double().mean().item()


if __name__ == "__main__":
    W0 = [th.tensor(w) for w in example()]
    scale = [w.pow(2).mean().sqrt() for w in W0]
    X, a, b = tie_lists_random() if "--random" in sys.argv else tie_lists()
    m0 = margins(W0, X, a, b); ms0 = m0.pow(2).mean()
    print(f"ARC: tie margin rms {ms0.sqrt():.3f}; accuracy {accuracy(W0):.4f}")
    R = th.randn(20_000, 10, generator=th.Generator().manual_seed(1)); S0 = scores(W0, R); ss0 = S0.pow(2).mean()
    lam = 0.1
    for mu in [float(v) for v in ([v for v in sys.argv[1:] if not v.startswith("--")] or ["100", "10", "1"])]:
        W = [w.clone().requires_grad_(True) for w in W0]
        opt = th.optim.Adam(W, lr=1e-3)
        for step in range(3000):
            opt.zero_grad()
            m = margins(W, X, a, b)
            dev = sum(((w - w0) / s).pow(2).sum() for w, w0, s in zip(W, W0, scale)) / sum(w.numel() for w in W0)
            fun = (scores(W, R) - S0).pow(2).mean() / ss0
            loss = m.pow(2).mean() / ms0 + mu * fun + lam * dev
            loss.backward(); opt.step()
        with th.no_grad():
            m = margins(W, X, a, b)
            rel = [((w - w0).norm() / w0.norm()).item() for w, w0 in zip(W, W0)]
            fun = ((scores(W, R) - S0).pow(2).mean() / ss0).sqrt().item()
            print(f"mu {mu:g}: score change on random lists {fun:.3f} of their size; tie margin rms {m.pow(2).mean().sqrt():.3f} ({(m.pow(2).mean() / ms0).sqrt():.2f} of ARC's); "
                  f"relative weight change input {rel[0]:.3f}, recurrent {rel[1]:.3f}, output {rel[2]:.3f}; accuracy {accuracy(W):.4f}")
            th.save([w.detach() for w in W], f"arc_star_mu{mu:g}{'_random' if '--random' in sys.argv else ''}.pt")
