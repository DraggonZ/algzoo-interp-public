"""Hand-built w7b (wider track, up to 24 neurons): w6b_relay's network (handbuilt_w6b, 22 neurons) with the hiding wave's
sign flip (a 180-degree part, one neuron) replaced by a second turning pair at 135 degrees (three neurons), so that the
memory turns votes in two modes (research/20261017_handbuilt/faithful_twin/LEDGER.md, branch w7b).

The wave. In the twin (handbuilt_v23.wave_coef) the hiding wave per unit of K d at offset k from the largest is
    f(k) = a cos(45 k) + b sin(45 k) + c (-1)^k,   f(0) = 1, f(1) = f(2) = 0  ->  a = 0.707, b = -0.293, c = 0.293.
Here the flip part is replaced by a pair turning 135 degrees per step (ARC's third rotation is about 127 degrees):
    f(k) = a cos(45 k) + b sin(45 k) + p cos(135 k) + q sin(135 k).
f(0) = 1, f(1) = f(2) = 0 leave one free coefficient (q = b, p = 1 - a, b = (1 - 2a) / 2). Every such wave has
f(+-4) = -1 and f(+-8) = 1 (both pairs are at -1 four steps on). Rule for the free coefficient: no spill at any other
offset in -9..9 (a spill is what an overtaken largest reveals). That gives a = p = 1/2, b = q = 0:
    f(k) = (cos(45 k) + cos(135 k)) / 2 = 1 at k = 0, +-8; -1 at k = +-4; 0 elsewhere.
(The twin's wave spills 0.41 and 0.59 at offsets -1, -2, -1 at 3, and 0.41-0.59 at 4-7.)

Build rule (build_w7b). Every weight of w6b is affine in the wave's coefficients for a fixed turn, and the 45- and
135-degree parts are separate linear systems that share only the comparator level. So:
  A = w6b built with the 45-degree clocks carrying (a, b) and no flip part (c = 0);
  B = w6b built with the clocks turning 135 degrees and carrying (p, q), no flip part;
  Z = w6b with no wave at all, from linearity: Z = 2 B(p, q) - B(2p, 2q) (checked against A: 2 A - A(2a, 2b) = Z).
  Neurons: A's, with A's clocks (45-degree pair) and B's clocks (135-degree pair, three new neurons) in place of the flip.
  Weights among the other neurons, and the readout of the other neurons: A + B - Z. Rows of the 45-degree clocks from
  A, of the 135-degree clocks from B. Every other neuron and the readout read the 45-degree clocks with A's weights and
  the 135-degree clocks with B's, keeping only the part that sums to zero over the three clocks (the wave: the clocks'
  level is read from C7 since w5b's level-free rule), and the flip is dropped: with c = 0 it holds only the level.
  Each part of every reader is the twin's rule for that part (hide at the readout, forward copy in the linear route,
  backward writes into the memory, the margin's deliveries from C1 and C7, the start relayed through H7), so the
  readers of the new wave follow the twin's rules for the new wave.
MAP (w7b_map): w6b's rows; slot 3 = the 45-degree clocks (as w6b, without the flip); the 135-degree clocks go to the
  memory slot among 12-15 whose ARC neuron takes the largest share of ARC's turning mode nearest 135 degrees
  (eigenvectors of ARC's recurrent matrix in its commonest on/off pattern, steps 4-8, as score_wide.py's job 9 reads
  them; an analysis read), and the neuron w6b put there becomes a helper.
Extra neurons over 16 (8): w6b's six (late gate, C7, C1, C6, CR, H7) and two for the second turning pair (three clocks
  in place of one flip neuron), for job 9 (and job 10, which needs job 9).
"""
import numpy as np
import algzoo.handbuilt_v23 as v23
from algzoo.handbuilt_w6b import build_w6b, w6b_map

TURN2 = 3 * np.pi / 4
COEF45 = (0.5, 0.0)
COEF135 = (0.5, 0.0)


def wave_new(k, c45=COEF45, c135=COEF135):
    return (c45[0] * np.cos(v23.TURN * k) + c45[1] * np.sin(v23.TURN * k)
            + c135[0] * np.cos(TURN2 * k) + c135[1] * np.sin(TURN2 * k))


def _build(turn, ab, **kw):
    """w6b with the clocks turning `turn` per step, carrying (a, b) of the wave and no flip part."""
    old_turn, old_coef = v23.TURN, v23.wave_coef
    v23.TURN = turn; v23.wave_coef = lambda *a_, **k_: (ab[0], ab[1], 0.0)
    try:
        wi, wh, wo, idx = build_w6b(**kw)
    finally:
        v23.TURN, v23.wave_coef = old_turn, old_coef
    return wi, wh, wo, idx


def build_w7b(c45=COEF45, c135=COEF135, check=True, **kw):
    A = _build(v23.TURN, c45, **kw)
    B = _build(TURN2, c135, **kw)
    B2 = _build(TURN2, (2 * c135[0], 2 * c135[1]), **kw)
    idx = dict(A[3]); n0 = len(A[0])
    Z = [2 * b - b2 for b, b2 in zip(B[:3], B2[:3])]
    if check:   # linearity: the no-wave network from A agrees with the one from B outside the clocks
        A2 = _build(v23.TURN, (2 * c45[0], 2 * c45[1]), **kw)
        ZA = [2 * a - a2 for a, a2 in zip(A[:3], A2[:3])]
        C = [idx[f"clock{k}"] for k in range(3)] + [idx["flip"]]
        oth = [k for k in range(n0) if k not in C]
        assert np.allclose(ZA[1][np.ix_(oth, oth)], Z[1][np.ix_(oth, oth)], atol=1e-9)
        assert np.allclose(ZA[2][:, oth], Z[2][:, oth], atol=1e-9)
    CA = [idx[f"clock{k}"] for k in range(3)]; P = idx["flip"]
    base = [k for k in range(n0) if k not in CA and k != P]
    # new layout: A's neurons with the flip's index taken by the first 135-degree clock, two more appended
    CB = [P, n0, n0 + 1]
    n = n0 + 2
    wi = np.zeros(n); wh = np.zeros((n, n)); wo = np.zeros((10, n))
    del idx["flip"]
    for k in range(3): idx[f"clock{k}"] = CA[k]; idx[f"turn{k}"] = CB[k]
    # weights among the other neurons, and their readout: A + B - Z
    for M_new, a_, b_, z_ in ((wh, A[1], B[1], Z[1]),):
        M_new[np.ix_(base, base)] = (a_ + b_ - z_)[np.ix_(base, base)]
    wo[:, base] = (A[2] + B[2] - Z[2])[:, base]
    wi[base] = A[0][base]
    assert np.allclose(A[0][base], B[0][base])
    zs = lambda w: w - w.mean(axis=-1, keepdims=True)     # the part of a read of three clocks that sums to zero
    for src, cl in ((A, CA), (B, CB)):
        wi_s, wh_s, wo_s, _ = src
        oc = CA   # the clocks' indices in the source build
        # rows of the clocks: from their own build (own clocks, other neurons; they read no flip)
        assert np.abs(wh_s[np.ix_(oc, [P])]).max() == 0
        wi[cl] = wi_s[oc]
        wh[np.ix_(cl, base)] = wh_s[np.ix_(oc, base)]
        wh[np.ix_(cl, cl)] = wh_s[np.ix_(oc, oc)]
        # other neurons and the readout read the wave only (their level read from these clocks plus the flip is 0)
        lv = wh_s[np.ix_(base, oc)].sum(1) + wh_s[base, P]
        assert np.abs(lv).max() < 1e-6, np.abs(lv).max()
        assert np.abs(wo_s[:, oc].sum(1) + wo_s[:, P]).max() < 1e-6
        wh[np.ix_(base, cl)] = zs(wh_s[np.ix_(base, oc)])
        wo[:, cl] = zs(wo_s[:, oc])
    return wi, wh, wo, idx


def arc_mode_slot(slots=(12, 13, 14, 15), target=135.0, n_lists=20000, seed=7):
    """The memory slot among `slots` whose ARC neuron takes the largest share of ARC's turning mode nearest `target`
    degrees, in ARC's commonest on/off pattern over steps 4-8 (score_wide.py's reading of job 9)."""
    import scipy.linalg as sl
    from collections import Counter
    from algzoo.core import example
    wi, wh, wo = [np.asarray(p, float) for p in example()]
    x = np.random.default_rng(seed).standard_normal((n_lists, 10)); h = np.zeros((n_lists, 16)); on = []
    for t in range(10):
        h = np.maximum(h @ wh.T + x[:, t:t + 1] * wi, 0); on.append(h > 0)
    on = np.stack(on, 1)[:, 4:9].reshape(-1, 16)
    top = np.frombuffer(Counter(map(lambda q: q.tobytes(), on)).most_common(1)[0][0], dtype=bool)
    ev, L, R = sl.eig(np.diag(top.astype(float)) @ wh, left=True, right=True)
    ks = [k for k in range(16) if ev[k].imag > 0]
    k = min(ks, key=lambda k: abs(np.degrees(np.angle(ev[k])) - target))
    lr = np.conj(L[:, k]) * R[:, k]; share = np.abs(lr) / np.abs(lr).sum()
    return int(max(slots, key=lambda j: share[j])), float(np.degrees(np.angle(ev[k]))), abs(ev[k]), share


def w7b_map(wi, wh, wo, idx, n_lists=20000, seed=0):
    import torch as th
    from algzoo.core import example
    CA = [idx[f"clock{k}"] for k in range(3)]; CB = [idx[f"turn{k}"] for k in range(3)]
    # w6b's map, read on the A-build layout (same neurons, flip -> first 135-degree clock)
    A = _build(v23.TURN, COEF45)
    M6, lay = w6b_map(*A)
    n = len(wi); M = np.zeros((16, n)); M[:, :M6.shape[1]] = M6
    M[:, CB] = 0; M[3] = 0
    j, _, _, _ = arc_mode_slot()
    M[j] = 0
    x = th.randn(n_lists, 10, generator=th.Generator().manual_seed(seed), dtype=th.float64).numpy()
    def acts(Wi, Wh):
        h = np.zeros((len(x), len(Wi))); out = []
        for t in range(10):
            h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi, 0); out.append(h)
        return np.stack(out[1:], 1).reshape(-1, len(Wi))
    E = [np.asarray(p, float) for p in example()]
    Aa = acts(E[0], E[1]); Ah = acts(wi, wh)
    rms = lambda v: np.sqrt((v ** 2).mean())
    M[3, CA] = rms(Aa[:, 3]) / rms(Ah[:, CA].sum(1))
    M[j, CB] = rms(Aa[:, j]) / rms(Ah[:, CB].sum(1))
    lay = dict(lay); lay[3] = "clock0+clock1+clock2"; lay[j] = "turn0+turn1+turn2"
    return M, lay
