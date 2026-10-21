"""Split each placement's second-third tie offset into exact parts (no biases: within one on/off region the margin is
exactly the sum of gradient x input). Margin = score[p_later] - score[p_earlier] at m2 = m3 = 0.9, m1 = 1.4, others -0.46.
Parts: vote mismatch (gradient on the later input + gradient on the earlier one, times 0.9), largest (gradient x 1.4),
background (sum of gradients x -0.46). Reports each part's spread across placements and its correlation with the total."""
import itertools
import sys

import numpy as np
import torch as th

from algzoo.core import example, zoo

T = np.array(list(itertools.permutations(range(10), 3)))
for name in sys.argv[1:] or ["example", 2, 3, 0, 4, 1]:
    P = [th.as_tensor(np.asarray(q), dtype=th.float64) for q in (example() if name == "example" else zoo(16, 10, int(name)))]
    w_hi, w_hh, w_oh = P
    X = th.full((len(T), 10), -0.46, dtype=th.float64)
    ar = th.arange(len(T))
    p1, p2, p3 = (th.as_tensor(T[:, i]) for i in range(3))
    X[ar, p1] = 1.4; X[ar, p2] = 0.9; X[ar, p3] = 0.9
    X.requires_grad_(True)
    h = th.zeros(len(T), 16, dtype=th.float64)
    for t in range(10):
        h = th.relu(h @ w_hh.T + X[:, t, None] * w_hi)
    s = h @ w_oh.T
    late = th.maximum(p2, p3); early = th.minimum(p2, p3)
    M = s[ar, late] - s[ar, early]
    g, = th.autograd.grad(M.sum(), X)
    g = g.numpy(); Xn = X.detach().numpy(); Mn = M.detach().numpy()
    ia = np.arange(len(T))
    gl, ge, g1 = g[ia, late.numpy()], g[ia, early.numpy()], g[ia, T[:, 0]]
    vote = (gl + ge) * 0.9
    big = g1 * 1.4
    bg = (g * Xn).sum(1) - vote - big
    assert np.allclose((g * Xn).sum(1), Mn, atol=1e-8)
    out = [f"{name}: tie margin (later minus earlier) mean {Mn.mean():+.2f}, spread {Mn.std():.2f}; later-vote slope median {np.median(gl):.0f}, earlier {np.median(-ge):.0f}"]
    for lab, v in [("vote mismatch", vote), ("largest", big), ("background", bg)]:
        out.append(f"{lab}: mean {v.mean():+.2f}, spread {v.std():.2f}, corr with margin {np.corrcoef(v, Mn)[0, 1]:+.2f}")
    print("; ".join(out), flush=True)


def robust(v):
    q1, q3 = np.percentile(v, [25, 75])
    return (q3 - q1) / 1.349


if __name__ == "__main__" and "--robust" in sys.argv:
    pass
