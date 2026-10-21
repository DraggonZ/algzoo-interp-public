"""Hand-built v5a: r4a's network (handbuilt_v4a: 2-D growing hiding wave, runner-up comparator, keep cut against it,
late gate writing the backward wave, optional memory ring) with two changes of layout and scale, no change of rule.
Research notes: research/20261017_handbuilt/faithful_twin/LEDGER.md, branch r5a.

to_arc(net, scale=...):
  "rms" (r4a's rule): each neuron scaled so that its RMS activity equals its ARC partner's.
  "ls": each neuron scaled by the least-squares factor k = <ours, ARC> / <ours, ours> over the same lists and steps,
        the single number that brings its values closest to its partner's. Still one number per neuron read off ARC's
        activity; rescaling a neuron and dividing its outgoing weights changes no answer.
"""
import numpy as np
from algzoo.handbuilt_v4a import build_v4a, LAYOUT  # noqa: F401  (build_v4a re-exported for the candidates)


def acts(W, x):
    import torch as th
    from algzoo.core import forward
    return forward(W, x, return_hidden=True)[1][:, 1:].reshape(-1, 16).numpy()


def to_arc(net, free=("mem1", "mem2", "mem5", "linear"), slots=(12, 13, 14, 15), n=20000, seed=0, scale="rms"):
    """Relabel into ARC's slots (LAYOUT; the four memory neurons without a switching job go to 12-15 by one-to-one
    activity correlation, match_twin.py's rule) and scale each neuron to its ARC partner (rule `scale`)."""
    import torch as th
    from scipy.optimize import linear_sum_assignment
    from algzoo.core import example
    H = [np.asarray(p, float) for p in net.params]
    Ar = [np.asarray(p, float) for p in example()]
    x = th.randn(n, 10, generator=th.Generator().manual_seed(seed), dtype=th.float64)
    lay = dict(LAYOUT)
    Aa, Ah = acts(Ar, x), acts(H, x)
    ours = [net.idx[k] for k in free]
    Cm = np.array([[abs(np.corrcoef(Ah[:, i], Aa[:, j])[0, 1]) for j in slots] for i in ours])
    r, c = linear_sum_assignment(-np.nan_to_num(Cm))
    names = {v: k for k, v in net.idx.items()}
    for a_, b_ in zip(r, c):
        lay[slots[b_]] = names[ours[a_]]
    P = np.array([net.idx[lay[j]] for j in range(16)])
    M = (H[0][P], H[1][np.ix_(P, P)], H[2][:, P])
    Am = acts(M, x)
    if scale == "rms":
        k = np.sqrt((Aa ** 2).mean(0)) / np.maximum(np.sqrt((Am ** 2).mean(0)), 1e-12)
    elif scale == "ls":
        k = (Am * Aa).sum(0) / np.maximum((Am ** 2).sum(0), 1e-300)
        k = np.where(k > 0, k, np.sqrt((Aa ** 2).mean(0)) / np.maximum(np.sqrt((Am ** 2).mean(0)), 1e-12))
    else:
        raise ValueError(scale)
    return M[0] * k, M[1] * k[:, None] / k[None, :], M[2] / k[None, :]
