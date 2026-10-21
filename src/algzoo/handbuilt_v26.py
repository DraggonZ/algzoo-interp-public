"""Hand-built v26: v25 (the twin + switch-off + keep cut) with options tried in round 2 of the faithful-twin work
(research/20261017_handbuilt/faithful_twin/LEDGER.md, branch r2b).

Used by the best candidate (candidates/r2b_loopL.py): build_v26() at its defaults (delta 0, kappa 0.5) is exactly
v25's network; then two changes of basis that leave the function unchanged wherever the neurons involved are on:
  mix_memory: ARC's slot 3 holds delay cell 6 plus delay cell 3 (cell 3 carries the gap at ages 1-2 through the
      detector's backward wave copy; cells 3 and 4 are also the age-5 and age-6 switchers, so the gap needed a second
      home). Its readers subtract the cell-3 part.
  loop_through_linear: the linear route (slot 9) holds L + nu x delay cell 7; its readers subtract nu x cell 7. The
      memory block alone then has a loop cell 7 -> cell 0 of weight nu (turning modes at 45, 90, 135 degrees), which
      the path through L cancels while L and cell 7 are on. Where either is clipped the loop acts (0.65% of answers
      change, accuracy +0.4 points).
Both pass checklist tests (neuron 3 carries the gap; memory turns votes) without changing how the network computes,
so they are not ARC's mechanism for those jobs.

Tried and dropped (change 4 below): a hiding wave whose forward spill is slightly negative, so that a delay cell
without a switching job carries the gap.

Base: `handbuilt_v25.build_v25` (docstring there; its own base is `handbuilt_v23.build_v23`, which explains the wave).

Change 4, negative forward spill (delta). The hiding wave per unit of K d is f(k) = a cos(45 k) + b sin(45 k) + c (-1)^k,
k = offset from the largest. The twin fixes f(0) = 1 and f(1) = f(2) = 0. Here f(1) = f(2) = -delta. Solving:
c = (cos45 (1 - delta) + delta) / (1 + 2 cos45), a = 1 - c, b = -delta - c.
Why it is safe: after a largest is overtaken its wave is revealed; the positions after it and before the overtaking
number hold numbers below it and can never be the answer, so lowering them changes no answer (a positive spill there
is what hurts). The backward spill grows with delta: f(-2) = delta + 2c reaches 1/K (K = 1.5) at delta = 0.065; the
twin's rule keeps backward spill under 1/K, so delta = 0.05 (a round value under that limit).
Why it is wanted: one step after a record the detector writes -K f(k) d into the memory cell for offset k (sign -1
storage). The wave has an 8-step period, so delay cell 6, which holds offsets -7 and -6 at ages 1 and 2, receives
-K f(1) d and -K f(2) d = +K delta d: the gap, with the largest's sign positive and the previous largest's negative.
With the twin's wave only delay cells 3 and 4 carry the gap, and those are the cells that switch off at ages 5 and 6.

Layout: as v25 (store 2, detector 4, keep 5, linear route 9, flip 1, clocks 7/6/8, delay cells 0, 3, 4 -> ARC 0, 11, 10),
plus delay cell 6 -> ARC 3; delay cells 1, 2, 5, 7 -> ARC 12-15 by one-to-one activity correlation.
"""
import contextlib
import numpy as np
import algzoo.handbuilt_v23 as v23
from algzoo.handbuilt_v25 import build_v25, LAYOUT  # noqa: F401  (build_v25: the same network at delta 0, kappa 0.5)

C45 = np.cos(np.pi / 4)


def coef_neg_spill(delta):
    c = (C45 * (1 - delta) + delta) / (1 + 2 * C45)
    return 1 - c, -delta - c, c


@contextlib.contextmanager
def wave_shape(coef):
    """build_v23 reads its wave coefficients from v23.wave_coef at call time; substitute ours for one build."""
    old = v23.wave_coef
    v23.wave_coef = lambda *a, **k: coef
    try:
        yield
    finally:
        v23.wave_coef = old


def build_v26(delta=0.0, eps=0.1, theta=0.75 * 0.58, kappa=0.5, a=1000.0, s=20.0, share=0.25):
    """v25's two edits (switch-off eps, keep cut theta; rules in handbuilt_v25) on v23 with wave spill delta and
    hiding margin kappa (the largest is hidden kappa x its jump below the previous largest)."""
    with wave_shape(coef_neg_spill(delta)):
        net = v23.build_v23(w=0.0, U_mem=1.0, share=share, rho=0.99, level_exact=False, a=a, s=s, kappa=kappa)
    wi, wh, wo = net.params
    S, D, G, L = (net.idx[k] for k in ("store", "detect", "keep", "linear"))
    wh[L, D] += eps / a
    beta = share * 1.0
    for M_, sg in ((G, -1.0), (L, 1.0)):
        wh[M_, S] += sg * beta * theta / s
        wh[M_, D] += sg * beta * theta / a
    return net


def mix_memory(net, into="mem6", add="mem3", mu=1.0):
    """Change of basis for one memory neuron: new value = cell `into` + mu x cell `add` (pre-activation), with every
    reader of it subtracting mu x its reading from cell `add` (W -> T W T^-1, input T wi, readout wo T^-1). Exact while
    both cells are on; when `add` is clipped, the mixed neuron keeps mu x its negative pre-activation one step."""
    import copy
    net = copy.deepcopy(net)
    wi, wh, wo = net.params
    i, j = net.idx[into], net.idx[add]
    T = np.eye(len(wi)); T[i, j] = mu
    Ti = np.eye(len(wi)); Ti[i, j] = -mu
    wh[:] = T @ wh @ Ti
    wi[:] = T @ wi
    wo[:] = wo @ Ti
    return net


def loop_through_linear(net, nu):
    """Change of basis for the linear route: new value = L + nu x delay cell 7. Its readers (memory cell 0 and the
    readout at position 8) subtract nu x their reading of L from cell 7 directly. The memory block then holds a loop
    cell 7 -> cell 0 of weight nu x (minus cell 0's weight on L) = nu, cancelled on clean inputs by the path through L.
    Exact while L and cell 7 are both on."""
    import copy
    net = copy.deepcopy(net)
    wi, wh, wo = net.params
    i, j = net.idx["linear"], net.idx["mem7"]
    T = np.eye(len(wi)); T[i, j] = nu
    Ti = np.eye(len(wi)); Ti[i, j] = -nu
    wh[:] = T @ wh @ Ti
    wi[:] = T @ wi
    wo[:] = wo @ Ti
    return net


def to_arc(net, fixed=None, free=(1, 2, 5, 7), slots=(12, 13, 14, 15), n=20000, seed=0):
    """Relabel into ARC's slots and scale each neuron to ARC's partner's RMS activity (as v25.to_arc).
    fixed: extra {ARC slot: v23 name}; free delay cells go to the free slots by activity correlation."""
    import torch as th
    from scipy.optimize import linear_sum_assignment
    from algzoo.core import example, forward
    H = [np.asarray(p, float) for p in net.params]
    A = [np.asarray(p, float) for p in example()]
    x = th.randn(n, 10, generator=th.Generator().manual_seed(seed), dtype=th.float64)
    acts = lambda W: forward(W, x, return_hidden=True)[1][:, 1:].reshape(-1, 16).numpy()
    lay = {k: v for k, v in LAYOUT.items() if k not in (3, 12, 13, 14, 15)}
    lay.update({3: "mem6"} if fixed is None else fixed)
    Aa, Ah = acts(A), acts(H)
    ours = [net.idx[f"mem{k}"] for k in free]
    C = np.array([[abs(np.corrcoef(Ah[:, i], Aa[:, j])[0, 1]) for j in slots] for i in ours])
    r, c = linear_sum_assignment(-C)
    names = {v: k for k, v in net.idx.items()}
    for a_, b_ in zip(r, c): lay[slots[b_]] = names[ours[a_]]
    P = np.array([net.idx[lay[j]] for j in range(16)])
    M = (H[0][P], H[1][np.ix_(P, P)], H[2][:, P])
    rms = lambda W: np.sqrt((acts(W) ** 2).mean(0))
    k = rms(A) / np.maximum(rms(M), 1e-12)
    return M[0] * k, M[1] * k[:, None] / k[None, :], M[2] / k[None, :]
