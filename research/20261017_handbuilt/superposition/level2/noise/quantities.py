"""Explanation quantities computed from a network's weights (batched over noisy copies).

Roles are given as a dict (ARC: store 2, detector 4, hiding 1 6 7 8, keep gate 5, late gate 9, memory the other eight).
Two kinds of quantity:
 - closed form in the weights: eigenvalues of the hiding block and the memory block, comparator gains (minus input
   weights), the store's leak, ratio and detector threshold, the store's share of each comparator's reference, the late
   gate's weights;
 - comparator thresholds as weight ratios with a fixed weighting of the sources: ref_n = W_hh[n, :] . hbar_n / c_n, where
   hbar_n is the clean network's mean state per unit of the current largest at the steps where comparator n is close
   to switching (its arriving number within 10% of its reference). The weighting comes from the clean network once;
   only the copy's weights change. The same with hbar taken over all steps gives each neuron's level per unit of the
   largest.
"""
import numpy as np
import torch as th
from noiselib import DEV, DT, flat, unflat

ARC_ROLES = {"store": 2, "detector": 4, "hiding": [1, 6, 7, 8], "keep": 5, "late": 9,
             "memory": [0, 3, 10, 11, 12, 13, 14, 15]}


def clean_states(W, n=200_000, seed=7):
    """States of the clean network: h(t-1), pre(t), x_t, running largest m1 (floored at 0, including x_t's predecessor)."""
    g = th.Generator(device=DEV).manual_seed(seed)
    X = th.randn(n, 10, device=DEV, dtype=DT, generator=g)
    wi, wh, wo = W; h = th.zeros(n, 16, device=DEV, dtype=DT); Hp, PRE = [], []
    for t in range(10):
        pre = h @ wh.T + X[:, t:t + 1] * wi; Hp.append(h); PRE.append(pre); h = th.relu(pre)
    m1 = th.cummax(th.cat([th.zeros(n, 1, device=DEV, dtype=DT), X], 1), 1).values[:, :-1]   # largest before x_t
    return X, th.stack(Hp, 1), th.stack(PRE, 1), m1


def weightings(W, roles, n=200_000):
    """Fixed source weightings from the clean network (per unit of the largest so far)."""
    X, Hp, PRE, m1 = clean_states(W, n)
    wi = W[0]; out = {}
    ok = (m1 > 0.3) & (th.arange(10, device=DEV)[None] >= 1)
    hr = Hp / m1.clamp(min=1e-9)[..., None]                                   # state per unit of the largest
    for nrn in roles["hiding"]:
        c = -wi[nrn]; ref = (PRE[:, :, nrn] + c * X) / c                       # reference = drive / c
        near = ok & ((X - ref).abs() < 0.1 * m1)
        out[f"near{nrn}"] = hr[near].mean(0)
    out["all"] = hr[ok].mean(0)
    return out


def eig_pairs(A):
    """Eigenvalues of a batch of square blocks, as (size, |angle| in degrees), sorted by size descending."""
    ev = th.linalg.eigvals(A)                                                  # [C, k]
    sz = ev.abs(); ang = th.angle(ev).abs() * 180 / np.pi
    o = th.argsort(-sz, 1)
    return th.gather(sz, 1, o), th.gather(ang, 1, o)


def quantities(W, dws, roles=ARC_ROLES, wts=None):
    """Dict name -> numpy [C] for copies W + dw."""
    wi, wh, wo = unflat(flat(W)[None] + dws, W)
    S, D, H, L, K, Mm = roles["store"], roles["detector"], roles["hiding"], roles["late"], roles["keep"], roles["memory"]
    q = {}
    hb = wh[:, H][:, :, H]
    sz, ang = eig_pairs(hb)
    q["hiding: shrink of the rotating pair"] = sz[:, 0]; q["hiding: rotation angle"] = ang[:, 0]
    q["hiding: size of the dying pair"] = sz[:, -1]
    for n in H: q[f"comparator {n}: gain"] = -wi[:, n]
    q["store: leak (self-weight)"] = wh[:, S, S]
    ratio = wh[:, S, D] * wi[:, D]; q["store: ratio per unit of the largest"] = ratio
    q["detector: threshold (fraction of the stored largest)"] = -wh[:, D, S] * ratio / wi[:, D]
    q["store: residue of the old largest after a jump"] = wh[:, S, S] + wh[:, S, D] * wh[:, D, S]
    for n in H: q[f"comparator {n}: store's share of reference"] = wh[:, n, S] * ratio / (-wi[:, n])
    msz, mang = eig_pairs(wh[:, Mm][:, :, Mm])
    for i in range(len(Mm)): q[f"memory: eigenvalue {i + 1} size"] = msz[:, i]; q[f"memory: eigenvalue {i + 1} angle"] = mang[:, i]
    if L is not None:
        q["late gate: off-switch from the largest's comparator"] = wh[:, L, H[2]] if len(H) > 2 else wh[:, L, H[0]]
        for n in H: q[f"late gate: push on comparator {n}"] = wh[:, n, L] / (-wi[:, n])
        q["late gate: memory read (sum of weights)"] = wh[:, L][:, Mm].sum(1)
    if wts is not None:
        for n in H:
            q[f"comparator {n}: threshold near switching"] = (wh[:, n] @ wts[f"near{n}"]) / (-wi[:, n])
            q[f"comparator {n}: level per unit of largest"] = (wh[:, n] @ wts["all"]) / (-wi[:, n])
        for n in Mm: q[f"memory {n}: level per unit of largest"] = wh[:, n] @ wts["all"]
        q["store: level per unit of largest"] = wh[:, S] @ wts["all"]
    return {k: v.detach().cpu().numpy() for k, v in q.items()}
