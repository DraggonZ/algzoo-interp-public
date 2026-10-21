"""Hand-built v25: the v23 twin with two changes toward ARC's memory jobs, laid out in ARC's neuron order.

Base: `handbuilt_v23.build_v23(U_mem=1, share=0.25, rho=0.99, level_exact=False)` (the twin; its docstring gives every
rule). Its memory is a delay line: memory cell j holds the number that arrived j + 2 steps ago, stored as the number's
shortfall below the largest so far. The cell that holds the current largest therefore holds about zero.

Change 1, switch-off (eps): one step after a record the detector adds eps x its jump d to the linear route
(weight detector -> linear route + eps / a). The memory stores minus the linear route, so two steps after the record
the cell that receives the largest gets - eps d and switches off. Zero is what the readout needed from that cell, so
clipping costs nothing while the largest stands; as the cell shifts down the line it stays off (it receives zero), so
the largest's cell is off at every age from 2 to 9. Cells 0, 3 and 4 hold the largest at ages 2, 5 and 6, as ARC's
memory neurons 0, 11 and 10 switch off at those ages. Side effect: a record at step 8 is read at position 8 straight from
the linear route, hidden by kappa - eps instead of kappa (kappa = 0.5).

Change 3, keep gate's cut (theta): the twin's keep gate passes every positive number (beta relu(x)). Here it passes a
number only above theta x the largest so far (the twin has no runner-up; the store has the largest):
G = beta relu(x - theta m1), with m1 the largest including x (store / s + detector / a), so a record always passes.
The linear route adds beta theta m1, so a vote is alpha x + beta max(x, theta m1): numbers below the cut get a floor
that rises with the largest at their arrival, instead of a penalty (an ideal vote model, r1a/a3_keep_ideal.py, loses
about 1 point with the floor and 8 with the penalty). Rule for theta: ARC's cut, 0.75 of the runner-up so far
(gates/README.md), times the median runner-up / largest so far over steps 2-8 (0.58, a property of normal lists,
r1a/a4_ratio.log): theta = 0.44. The last number is read through the direct route only and gets no floor.

Change 2, loop (w): v23's own option, last memory cell back into the first with weight w (with v23's compensations).
The eight cells then turn by 0, 45, 90, 135, 180 degrees per step, size (w rho^7)^(1/8).

Layout in ARC's order (relabelling and rescaling only, the same function): store 2, detector 4, keep gate 5, linear
route 9 (ARC's late-gate slot; there is no late gate), flip 1, clock 0/1/2 -> 7/6/8 (as match_twin.py found),
memory cell 0 -> 0, cell 3 -> 11, cell 4 -> 10 (switching ages 2, 5, 6), cells 1, 2, 5, 6, 7 -> 15, 3, 13, 14, 12
(cells 1, 2, 6 kept where match_twin.py put them; 5 and 7 take the two slots left).
Each neuron's loudness is set to ARC's partner's (RMS of activity on 20,000 lists, seed 0): 16 scale constants read
off ARC's activity, which change no decision.
"""
import numpy as np
from algzoo.handbuilt_v23 import build_v23

# ARC slot -> v23 neuron name
LAYOUT = {2: "store", 4: "detect", 5: "keep", 9: "linear", 1: "flip", 7: "clock0", 6: "clock1", 8: "clock2",
          0: "mem0", 11: "mem3", 10: "mem4", 15: "mem1", 3: "mem2", 13: "mem5", 14: "mem6", 12: "mem7"}


def build_v25(eps=0.0, w=0.0, theta=0.0, a=1000.0, s=20.0, share=0.25):
    net = build_v23(w=w, U_mem=1.0, share=share, rho=0.99, level_exact=False, a=a, s=s)
    wi, wh, wo = net.params
    S, D, G, L = (net.idx[k] for k in ("store", "detect", "keep", "linear"))
    wh[L, D] += eps / a
    # keep gate cut (theta): G = beta relu(x - theta m1), m1 the largest after x (store/s + detector/a);
    # the linear route adds beta theta m1, so each vote is alpha x + beta max(x, theta m1) (a floor, not a penalty)
    beta = share * 1.0
    for M_, sg in ((G, -1.0), (L, 1.0)):
        wh[M_, S] += sg * beta * theta / s
        wh[M_, D] += sg * beta * theta / a
    return net


def to_arc(net, n=20000, seed=0, free_by_corr=False):
    """Relabel into ARC's slots (LAYOUT) and scale each neuron to ARC's partner's RMS activity.
    free_by_corr: the five memory cells without a switching job (1, 2, 5, 6, 7) go to ARC's slots 3, 12, 13, 14, 15 by
    one-to-one assignment on activity correlation (as match_twin.py did for all eight), instead of LAYOUT's fixed choice."""
    import torch as th
    from algzoo.core import example, forward
    H = [np.asarray(p, float) for p in net.params]
    x = th.randn(n, 10, generator=th.Generator().manual_seed(seed), dtype=th.float64)
    A = [np.asarray(p, float) for p in example()]
    acts = lambda W: forward(W, x, return_hidden=True)[1][:, 1:].reshape(-1, 16).numpy()
    lay = dict(LAYOUT)
    if free_by_corr:
        from scipy.optimize import linear_sum_assignment
        Aa, Ah = acts(A), acts(H)
        ours = [net.idx[f"mem{k}"] for k in (1, 2, 5, 6, 7)]; theirs = [3, 12, 13, 14, 15]
        C = np.array([[abs(np.corrcoef(Ah[:, i], Aa[:, j])[0, 1]) for j in theirs] for i in ours])
        r, c = linear_sum_assignment(-C)
        names = {v: k for k, v in net.idx.items()}
        for a_, b_ in zip(r, c): lay[theirs[b_]] = names[ours[a_]]
    P = np.array([net.idx[lay[j]] for j in range(16)])
    M = (H[0][P], H[1][np.ix_(P, P)], H[2][:, P])
    rms = lambda W: np.sqrt((acts(W) ** 2).mean(0))
    c = rms(A) / np.maximum(rms(M), 1e-12)
    return M[0] * c, M[1] * c[:, None] / c[None, :], M[2] / c[None, :]
