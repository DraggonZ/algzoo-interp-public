"""Hand-built v9a: the hiding wave moved out of the hiding slots by a two-set relay.
Research notes: research/20261017_handbuilt/faithful_twin/LEDGER.md, branch r9a.

Base: r6a_gap's network (handbuilt_v6a). Its three clocks C hold the comparator level gc (m - x) on the shared
direction and the 2-D hiding wave z on the other two (C = level x 1 + FROM_PQ z). The relay adds a set R of the same
kind, and each set reads the wave only from the other:
    C(t+1) = relu(gc (m_t - x_{t+1}) x 1 + FROM_PQ A z(R_t))
    R(t+1) = relu(gc (m_t - x_{t+1}) x 1 + F A z(C_t) + F b d_t)      (the record's start goes into R only)
so the wave lives in R one step after a record, in C the step after, and so on; the other set holds the level alone.
Both sets clip at a record (same comparator level), so the wave is wiped as before. Every other reader of the wave
(linear route copy, readout hiding, ring compensation) reads z(C) + z(R): a row's weights w on C are split into the
level read (w . 1) and the wave read u = w FROM_PQ; the wave read is added on R.
  n_relay = 3: F = FROM_PQ, z(R) = TO_PQ R.
  n_relay = 2: F = the first two rows of FROM_PQ, R = level x 1 + F z, decoded as z(R) = F^-1 (R - level x 1) with the
               level read from C (lv . C, lv = the first row of PHI^-1; exact because C's wave is on FROM_PQ).
The network's function is unchanged (exactly, while no clock or relay clips part-way). R is meant for the hiding
slots 1, 6, 7: a nudge to R is gone one step later (R is rebuilt from C), and silencing R one step after the largest
removes the wave that hides it.
"""
import numpy as np
from algzoo.handbuilt import FROM_PQ, TO_PQ, PHI
from algzoo.handbuilt_v4a import wave_2d
from algzoo.handbuilt_v6a import build_v6a


def add_relays(params, C, D, n_relay=3, gc=1e10, a=1000.0, kappa=0.5, level_from_relays=()):
    """params of a v4a-family network (own order) -> params with n_relay relay neurons appended (indices n..n+k-1)."""
    wi0, wh0, wo0 = (np.asarray(p, float) for p in params)
    n = len(wi0); k = n_relay; N = n + k; R = list(range(n, N)); C = list(C)
    wi = np.zeros(N); wh = np.zeros((N, N)); wo = np.zeros((wo0.shape[0], N))
    wi[:n] = wi0; wh[:n, :n] = wh0; wo[:, :n] = wo0
    Aw, bw, _ = wave_2d(K=1 + kappa)
    lv = np.linalg.inv(PHI)[0]                       # level from C: lv . 1 = 1, lv . FROM_PQ = 0
    if k == 3:
        F = FROM_PQ; dec_R = TO_PQ; dec_C_level = np.zeros(2)   # z(R) = TO_PQ R
    else:
        F = FROM_PQ[:2]; Fi = np.linalg.inv(F)
        dec_R = Fi; dec_C_level = -(Fi @ np.ones(2))           # z(R) = Fi R - (Fi 1) level(C)
    start = wh0[np.ix_(C, [D])][:, 0] - gc / a * np.ones(3)     # FROM_PQ b / a, the record's start in C's rows
    assert np.allclose(start, FROM_PQ @ bw / a)
    # relay rows: level as C's (store, detector, input), wave read from C, the start
    lvl_rows = wh0[C].copy(); lvl_rows[:, C] = 0.0; lvl_rows[:, D] = gc / a
    assert np.allclose(lvl_rows, lvl_rows[0]), "C's level rows differ"
    for i, r in enumerate(R):
        wh[r, :n] = lvl_rows[0]
    wi[R] = wi0[C[0]]
    wh[np.ix_(R, C)] = F @ Aw @ TO_PQ
    wh[R, D] += F @ bw / a
    # clocks: no start, wave read from R instead of from themselves
    wh[C, D] -= FROM_PQ @ bw / a
    wh[np.ix_(C, C)] = 0.0
    wh[np.ix_(C, R)] = FROM_PQ @ Aw @ dec_R
    wh[np.ix_(C, C)] += np.outer(FROM_PQ @ Aw @ dec_C_level, lv)
    # every other reader: add the wave read on R
    for M, rows in ((wh, [j for j in range(n) if j not in C]), (wo, range(wo.shape[0]))):
        for j in rows:
            u = M[j, C] @ FROM_PQ
            if not np.any(u): continue
            M[j, R] += u @ dec_R
            M[j, C] += (u @ dec_C_level) * lv
    for j in level_from_relays:   # these rows read the level from the relays (the same value; relays' wave sums to 0)
        assert k == 3 and np.allclose(wh[j, C] @ FROM_PQ, 0)
        wh[j, R] += wh[j, C]; wh[j, C] = 0.0
    return wi, wh, wo


def relay_map(net, P, n_lists=20000, seed=0):
    """Wider-track MAP [16, 19] for the three-relay network (analysis only, not a 16-neuron candidate): v4a's layout
    with the relays in the hiding slots 1, 6, 7 (relay k mirrors clock k) and the clocks as helpers in no slot; the
    four memory-like neurons to 12-15 by activity correlation; RMS scale per row (handbuilt_w4b.arc_map's rules)."""
    idx = dict(net.idx); n = len(P[0])
    for k in range(n - 16): idx[f"relay{k}"] = 16 + k
    # handbuilt_w4b.arc_map's correlation and scale rules, with this layout
    import torch as th
    from scipy.optimize import linear_sum_assignment
    from algzoo.core import example
    x = th.randn(n_lists, 10, generator=th.Generator().manual_seed(seed), dtype=th.float64).numpy()
    def acts(Wi, Wh):
        h = np.zeros((len(x), len(Wi))); out = []
        for t in range(10):
            h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi, 0); out.append(h)
        return np.stack(out[1:], 1).reshape(-1, len(Wi))
    A = [np.asarray(p, float) for p in example()]
    Aa, Ah = acts(A[0], A[1]), acts(P[0], P[1])
    lay = {2: "store", 4: "detect", 5: "keep", 9: "late", 8: "runnerup", 1: "relay0", 6: "relay1", 7: "relay2",
           0: "mem0", 11: "mem3", 10: "mem4", 3: "mem6"}
    free = ["mem1", "mem2", "mem5", "linear"]; slots = [12, 13, 14, 15]
    Cm = np.array([[abs(np.corrcoef(Ah[:, idx[f]], Aa[:, j])[0, 1]) for j in slots] for f in free])
    r, c = linear_sum_assignment(-np.nan_to_num(Cm))
    for a_, b_ in zip(r, c): lay[slots[b_]] = free[a_]
    M = np.zeros((16, n))
    for j in range(16):
        k = idx[lay[j]]
        M[j, k] = np.sqrt((Aa[:, j] ** 2).mean()) / max(np.sqrt((Ah[:, k] ** 2).mean()), 1e-12)
    return M, lay


def build_v9a(n_relay=3, gap_from_relays=False, **kw):
    """gap_from_relays: neuron 3 (delay cell mem6) reads its gap from the hiding slots, the relays' level minus the
    runner-up comparator (as ARC's neuron 3 reads hiding 1 minus hiding 8), instead of the clocks' level."""
    net = build_v6a(ring_tap=5, gap_eff=1.0, last_comp=True, **kw)
    C = [net.idx[f"clock{k}"] for k in range(3)]
    rows = [net.idx["mem6"]] if gap_from_relays else []
    return net, add_relays(net.params, C, net.idx["detect"], n_relay=n_relay, level_from_relays=rows), C
